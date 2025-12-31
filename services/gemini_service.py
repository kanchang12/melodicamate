import json
import logging
import os
import base64
from typing import Dict, List

import google.generativeai as genai

logger = logging.getLogger("melodicamate.gemini")


class GeminiService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.config_error: str = ""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as exc:
                self.model = None
                self.config_error = f"Gemini init failed: {exc}"
                logger.error(self.config_error)
        else:
            self.model = None

    def _enabled(self) -> bool:
        return bool(self.api_key and self.model and not self.config_error)

    def generate_coaching_text(self, payload: Dict) -> str:
        """
        Generate warm, human coaching that focuses on musicality and song context,
        not technical jargon.
        """
        accuracy = payload.get("accuracy_pct", 0)
        lyric_context = payload.get("lyric_context") or []
        lyrics_lines = payload.get("lyrics_lines") or []
        title = payload.get("title") or "this song"
        
        if not self._enabled():
            return self._fallback_human_coaching(title, accuracy, lyric_context)
        
        try:
            prompt_parts = [
                "You are a warm, encouraging music teacher having a conversation with a student.",
                "Give friendly, practical advice focused on the musical experience.",
                "Keep your response conversational: 2-3 short sentences, under 280 characters total.",
                "",
                "RULES:",
                "- Talk about the song, the melody, the feel - NOT technical note names or numbers",
                "- Reference the lyrics or song sections when giving feedback",
                "- Use musical language: 'that upward jump', 'the chorus melody', 'where it goes higher'",
                "- Be encouraging but honest - suggest one concrete thing to work on",
                "- Never mention MIDI numbers, note names like 'B flat', or scale degrees",
                "- Sound like a human teacher, not a robot analyzer",
                "",
                f"Song: '{title}'",
                f"Performance accuracy: {accuracy:.0f}%",
            ]
            
            if lyric_context and accuracy < 95:
                problem_spots = []
                for word in lyric_context[:2]:
                    if word:
                        problem_spots.append(f"'{word}'")
                
                if problem_spots:
                    prompt_parts.append(f"Trouble spots near: {', '.join(problem_spots)}")
            
            if accuracy >= 90:
                prompt_parts.append("Performance was very strong, just minor polish needed.")
            elif accuracy >= 70:
                prompt_parts.append("Decent attempt with some pitch challenges to work on.")
            elif accuracy >= 50:
                prompt_parts.append("Several parts need work - focus on the melody shape and flow.")
            else:
                prompt_parts.append("Struggling with pitch accuracy - may need to slow down.")
            
            prompt_parts.append("")
            prompt_parts.append("Now give warm, practical feedback about this performance:")
            
            prompt = "\n".join(prompt_parts)
            
            resp = self.model.generate_content(prompt)
            
            if resp and resp.text:
                coaching = resp.text.strip()
                coaching = coaching.strip('"').strip("'")
                return coaching
            
            return self._fallback_human_coaching(title, accuracy, lyric_context)
            
        except Exception as exc:
            logger.warning("Gemini coaching fallback due to error: %s", exc)
            return self._fallback_human_coaching(title, accuracy, lyric_context)

    def _fallback_human_coaching(self, title: str, accuracy: float, lyric_context: List[str]) -> str:
        """
        Provide warm, human fallback coaching when AI is unavailable.
        """
        problem_words = []
        for word in (lyric_context or [])[:2]:
            if word:
                problem_words.append(f"'{word}'")
        
        spot_mention = f" especially around {' and '.join(problem_words)}" if problem_words else ""
        
        if accuracy >= 90:
            return f"Excellent work on {title}! You're really close{spot_mention}. One more smooth run-through and you've got it."
        
        if accuracy >= 75:
            return f"Good progress on {title}! The melody is taking shape{spot_mention}. Try singing it slower to nail those tricky spots."
        
        if accuracy >= 60:
            return f"Nice effort on {title}. Focus on following the melody's ups and downs{spot_mention}. Hum it first to get the feel."
        
        if accuracy >= 40:
            return f"Keep working on {title}! Try listening to it a few times first, then sing along. Focus on matching the melody shape."
        
        return f"Let's take {title} slower. Hum the melody first to learn the pattern, then add your voice. You'll get there!"

    def analyze_user_recording(self, audio_data: bytes, song_title: str = "Unknown Song") -> Dict:
        """
        Analyze a user's recorded audio (their own singing/humming) and extract the melody.
        
        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            song_title: Optional title for context
            
        Returns:
            Dict with melody as scale-degree numbers
        """
        if not self._enabled():
            return {"found": False, "notes": "Gemini disabled", "error": "gemini_failed"}

        try:
            # Convert audio to base64 for Gemini
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            prompt = f"""Analyze this audio recording and extract the melody as scale-degree numbers.

TASK:
The user has recorded themselves singing, humming, or whistling a melody.
Your job is to:
1. Listen to the audio
2. Identify the pitches being sung
3. Determine the key and mode
4. Convert the melody to scale-degree notation (1-7)

SCALE DEGREE NOTATION:
- 1 = tonic (root note)
- 2 = supertonic
- 3 = mediant
- 4 = subdominant
- 5 = dominant
- 6 = submediant
- 7 = leading tone
- Use 'b' for flats (b3, b7)
- Use '#' for sharps (#4, #5)

RESPONSE FORMAT (JSON):
{{
  "found": true,
  "confidence": "high|medium|low",
  "key": "C",
  "mode": "major",
  "notes": [1, 1, 5, 5, 6, 6, 5],
  "tempo_bpm": 120,
  "notes_text": "Analysis notes about the recording quality and any issues"
}}

Song context: {song_title}

IMPORTANT:
- Be honest about confidence - if the audio is unclear, set confidence to "low"
- Extract ONLY the sung melody, ignore background noise
- If you can't detect clear pitches, explain why in notes_text"""

            # Create the audio part for Gemini
            audio_part = {
                "mime_type": "audio/wav",  # Adjust based on actual format
                "data": audio_base64
            }
            
            response = self.model.generate_content([prompt, audio_part])
            
            if not response or not response.text:
                return {"found": False, "error": "empty_response"}

            text = response.text.strip()
            
            # Remove markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            # Print for debugging
            logger.info("=" * 80)
            logger.info(f"GEMINI AUDIO ANALYSIS FOR '{song_title}':")
            logger.info(text)
            logger.info("=" * 80)
            print("\n" + "=" * 80)
            print(f"GEMINI AUDIO ANALYSIS FOR '{song_title}':")
            print(text)
            print("=" * 80 + "\n")
            
            data = json.loads(text)
            
            # Validate response
            if not data.get("found"):
                return {
                    "found": False,
                    "error": "no_melody_detected",
                    "notes": data.get("notes_text", "Could not detect melody in recording")
                }
            
            notes = data.get("notes", [])
            if len(notes) < 3:
                logger.warning(f"Very short melody detected: {len(notes)} notes")
                data["notes_text"] = (data.get("notes_text", "") + 
                                     " [Warning: Very short melody detected]")
            
            return {
                "found": True,
                "confidence": data.get("confidence", "unknown"),
                "key": data.get("key", "C"),
                "mode": data.get("mode", "major"),
                "numbers": [str(n) for n in notes],  # Convert to strings for consistency
                "tempo_bpm": data.get("tempo_bpm", 100),
                "notes": data.get("notes_text", ""),
                "source": "user_recording"
            }
            
        except json.JSONDecodeError as exc:
            logger.error(f"Gemini returned invalid JSON: {exc}")
            return {"found": False, "error": "invalid_json", "notes": str(exc)}
        except Exception as exc:
            logger.error(f"Audio analysis failed: {exc}")
            return {"found": False, "error": str(exc)}

    def classify_song_request(self, query: str, composer: str = "") -> Dict:
        # Allow all; rely on Gemini to return numbers/lyrics. No PD gate here.
        return {
            "likely_public_domain": True,
            "canonical_title": query.title(),
            "composer": composer or "",
            "search_terms": [query],
            "notes": "Classification delegated to Gemini.",
        }

    def generate_song_numbers(self, query: str) -> Dict:
        """
        DEPRECATED: This method tries to recall songs from memory, which doesn't work well.
        Use analyze_user_recording() instead for user-provided audio.
        
        Kept for backward compatibility but will return low confidence.
        """
        if not self._enabled():
            return {"found": False, "notes": "Gemini disabled", "error": "gemini_failed"}

        return {
            "found": False,
            "confidence": "low",
            "notes": (
                "This method has been deprecated. "
                "Please use the recording feature where you sing/hum the song yourself. "
                "This ensures legal compliance and better accuracy."
            ),
            "error": "use_recording_instead"
        }
