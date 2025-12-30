import json
import logging
import os
from typing import Dict

import google.generativeai as genai

logger = logging.getLogger("melodicamate.gemini")


class GeminiService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.config_error: str = ""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as exc:  # pragma: no cover
                self.model = None
                self.config_error = f"Gemini init failed: {exc}"
                logger.error(self.config_error)
        else:
            self.model = None

    def _enabled(self) -> bool:
        return bool(self.api_key and self.model and not self.config_error)



    def classify_song_request(self, query: str, composer: str = "") -> Dict:
        # Allow all; rely on Gemini to return numbers/lyrics. No PD gate here.
        return {
            "likely_public_domain": True,
            "canonical_title": query.title(),
            "composer": composer or "",
            "search_terms": [query],
            "notes": "Classification delegated to Gemini.",
        }


    def generate_coaching_text(self, payload: Dict) -> str:
        """
        Generate warm, human coaching that focuses on musicality and song context,
        not technical jargon.
        """
        accuracy = payload.get("accuracy_pct", 0)
        mistakes = payload.get("mistakes", {})
        expected = payload.get("expected") or []
        played = payload.get("played") or []
        lyric_context = payload.get("lyric_context") or []
        title = payload.get("title") or payload.get("exercise_id", "this song")
        
        if not self._enabled():
            return self._fallback_human_coaching(title, accuracy, lyric_context)
        
        try:
            # Build a context-rich, musical prompt
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
            
            # Add context about where mistakes happened (lyrics-based)
            if lyric_context and accuracy < 95:
                # Just mention the first 2 problem spots
                problem_spots = []
                for ctx in lyric_context[:2]:
                    # Extract just the lyric word, not the technical stuff
                    if "'" in ctx:
                        # Extract text between quotes
                        lyric = ctx.split("'")[1] if "'" in ctx else ""
                        if lyric:
                            problem_spots.append(f"'{lyric}'")
                
                if problem_spots:
                    prompt_parts.append(f"Trouble spots near: {', '.join(problem_spots)}")
            
            # Add performance quality context
            if accuracy >= 90:
                prompt_parts.append("Performance was very strong, just minor polish needed.")
            elif accuracy >= 70:
                prompt_parts.append("Decent attempt with some pitch challenges to work on.")
            elif accuracy >= 50:
                prompt_parts.append("Several parts need work - focus on the melody shape and flow.")
            else:
                prompt_parts.append("Struggling with pitch accuracy - may need to slow down.")
            
            prompt_parts.extend([
                "",
                "EXAMPLE GOOD RESPONSES:",
                "- 'Nice work on Twinkle Twinkle! The opening was spot-on. Try smoothing out that upward jump in the second line - imagine reaching gently upward instead of jumping.'",
                "- 'Happy Birthday sounded great until the chorus. That high part on 'to you' needs more air support - take a bigger breath before it.'",
                "- 'You've got the rhythm of Amazing Grace down! Focus on the melody when you sing 'how sweet' - it climbs up then comes back down. Hum it a few times to feel the shape.'",
                "",
                "EXAMPLE BAD RESPONSES (never do this):",
                "- 'You played B flat instead of B natural at measure 3'",
                "- 'Scale degree 5 was incorrect, should be degree 6'",
                "- 'MIDI note 63 was detected instead of 65'",
                "",
                "Now give warm, practical feedback about this performance:",
            ])
            
            prompt = "\n".join(prompt_parts)
            
            resp = self.model.generate_content(prompt)
            
            if resp and resp.text:
                coaching = resp.text.strip()
                # Remove any quotes if the model wrapped the response
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
        # Extract just the lyric words from context (if any)
        problem_words = []
        for ctx in (lyric_context or [])[:2]:
            if "'" in ctx:
                word = ctx.split("'")[1] if len(ctx.split("'")) > 1 else ""
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

    def generate_song_numbers(self, query: str) -> Dict:
        """
        Use Gemini to return interleaved lyrics and scale-degree numbers.
        Returns a single formatted string in 'lyrics' for the Flutter UI.
        """
        if not self._enabled():
            return {"found": False, "notes": "Gemini disabled", "error": "gemini_failed"}

        prompt = (
            "You are a professional music transcription assistant with perfect pitch recall. "
            "Given a song title, return the EXACT melody as scale-degree numbers relative to the tonic.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Return the ACTUAL melody notes, not an approximation. Each number must match the real song.\n"
            "2. Scale degrees: 1=tonic, 2=supertonic, 3=mediant, 4=subdominant, 5=dominant, 6=submediant, 7=leading tone\n"
            "3. Use 'b' for flats (e.g., b3, b7) and '#' for sharps (e.g., #4) when needed\n"
            "4. Alternate lines: Line 1=Lyrics, Line 2=Numbers (one number per syllable), Line 3=Lyrics, Line 4=Numbers\n"
            "5. Match rhythm: Use dots (...) for held notes, dashes (---) for rests\n"
            "6. One number per sung syllable - align numbers with lyric syllables precisely\n"
            "7. Double-check your melody against the actual song before responding\n\n"
            "EXAMPLES:\n"
            "Twinkle Twinkle Little Star (C major):\n"
            "Lyrics: 'Twin-kle twin-kle lit-tle star'\n"
            "Numbers: '1 1 5 5 6 6 5...'\n\n"
            "Happy Birthday (C major):\n"
            "Lyrics: 'Hap-py birth-day to you'\n"
            "Numbers: '5 5 6 5 1 7...'\n\n"
            "Response format (valid JSON only):\n"
            "{\n"
            '  "found": true,\n'
            '  "confidence": "high|medium|low",\n'
            '  "key": "C",\n'
            '  "mode": "major",\n'
            '  "time_signature": "4/4",\n'
            '  "tempo_bpm": 90,\n'
            '  "lines": [\n'
            '    "Lyric line with syllables",\n'
            '    "1 1 5 5 6 6 5",\n'
            '    "Next lyric line",\n'
            '    "1 2 3 4 5"\n'
            '  ],\n'
            '  "notes": "Context about transcription accuracy and any uncertainties"\n'
            "}\n\n"
            "IMPORTANT: If you're not confident about the exact melody, set confidence to 'low' "
            "and explain in 'notes' what you're uncertain about. Do not guess.\n\n"
            f"Transcribe this song accurately: {query}\n\n"
            "Think step-by-step:\n"
            "1. Recall the exact melody from memory\n"
            "2. Identify the key and starting pitch\n"
            "3. Convert each note to its scale degree\n"
            "4. Verify it matches the actual song\n"
            "5. Return the JSON response"
        )

        try:
            resp = self.model.generate_content(prompt)
            if not resp or not resp.text:
                return {"found": False, "error": "empty_response"}

            text = resp.text.strip()
            # Remove markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            
            data = json.loads(text)

            # Check confidence level
            confidence = data.get("confidence", "unknown")
            if confidence == "low":
                logger.warning(
                    f"Low confidence transcription for '{query}': {data.get('notes', '')}"
                )

            raw_lines = data.get("lines", [])
            numbers_only = []

            # Interleave lyrics and numbers for display
            interleaved_string = "\n".join(raw_lines)

            # Extract just the numbers for playback
            for i, line in enumerate(raw_lines):
                if i % 2 != 0:  # Odd indices are number lines
                    # Clean up notation: remove dots and dashes, keep only numbers
                    cleaned = line.replace(".", " ").replace("-", " ")
                    numbers_only.extend([n for n in cleaned.split() if n and n[0].isdigit() or n[0] in ['b', '#']])

            result = {
                "found": bool(data.get("found")),
                "confidence": confidence,
                "key": data.get("key"),
                "mode": data.get("mode"),
                "time_signature": data.get("time_signature"),
                "tempo_bpm": data.get("tempo_bpm"),
                "lines": raw_lines,
                "numbers": numbers_only,
                "lyrics": interleaved_string,
                "notes": data.get("notes", ""),
            }
            
            # Log warning if transcription seems suspicious
            if len(numbers_only) < 8:
                logger.warning(f"Suspiciously short melody for '{query}': {len(numbers_only)} notes")
                result["notes"] += " [Warning: Very short melody - verify accuracy]"
            
            return result
            
        except json.JSONDecodeError as exc:
            logger.error(f"Gemini returned invalid JSON for '{query}': {exc}")
            logger.error(f"Raw response: {resp.text if resp else 'None'}")
            return {"found": False, "error": "invalid_json", "notes": str(exc)}
        except Exception as exc:
            logger.error("Gemini song lookup failed: %s", exc)
            return {"found": False, "error": str(exc)}

    def _fallback(self, base: str, accuracy: float) -> str:
        if accuracy >= 90:
            return "Nice work! You're very close—keep the airflow steady and repeat once more for consistency."
        if accuracy >= 70:
            return "Good effort. Watch the intonation on the missed notes and keep the tempo steady."
        return "Let's try again slower. Focus on matching each pitch; breathe evenly and aim for clean note starts."
