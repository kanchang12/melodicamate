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

    def generate_coaching_text(self, payload: Dict) -> str:
        accuracy = payload.get("accuracy_pct", 0)
        mistakes = payload.get("mistakes", {}).get("summary", "")
        base = (
            f"{payload.get('exercise_id')} in key {payload.get('key')} {payload.get('mode')} "
            f"accuracy {accuracy:.1f}%. {mistakes}"
        )
        if not self._enabled():
            return self._fallback(base, accuracy)
        try:
            prompt = (
                "You are a concise music coach. "
                "Reply in <=2 sentences, max 300 chars, actionable. "
                f"Summary: {base}"
            )
            resp = self.model.generate_content(prompt)
            return resp.text.strip() if resp and resp.text else self._fallback(base, accuracy)
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini coaching fallback due to error: %s", exc)
            return self._fallback(base, accuracy)

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
        Use Gemini to return the full primary melody (verse + chorus) as scale-degree numbers (relative to tonic) and lyrics snippet if known.
        Expected JSON (do not wrap in code fences):
        {
          "found": true/false,
          "key": "G",
          "mode": "major",
          "time_signature": "4/4",
          "tempo_bpm": 90,
          "measures": [["1","2","3","4"],["5","6","7","1"]],
          "numbers": ["1","2",...],  // flattened
          "lyrics": "snippet",
          "notes": "any notes"
        }
        numbers should cover the main melody (verse and chorus, ~32-96 notes) in order, with repeats included.
        """
        if not self._enabled():
            notes = self.config_error or "Gemini disabled"
            return {
                "found": False,
                "numbers": [],
                "lyrics": "",
                "notes": notes,
                "error": "gemini_failed",
            }
        prompt = (
            "You are a professional music transcription assistant. Given a song title (and optional artist), "
            "return the ACTUAL primary melody from the real song as scale-degree numbers (solfege) relative to the tonic.\n\n"
            "IMPORTANT RULES:\n"
            "1. Return the REAL, RECOGNIZABLE melody from the actual song - NOT random notes\n"
            "2. Include verse AND chorus (aim for 32-96 notes total with repeats)\n"
            "3. Organize into MEASURES based on time signature (typically 4 beats per measure in 4/4 time)\n"
            "4. Use appropriate tempo for the song (e.g., 60-80 bpm for ballads, 100-140 for upbeat)\n"
            "5. Identify the correct key and mode (major/minor) for the song\n"
            "6. For rhythm: use single notes per beat OR break beats into smaller durations\n"
            "7. Include a short lyrics snippet from the song if known\n\n"
            "Response format (JSON only, no code fences):\n"
            "{\n"
            "  \"found\": true/false,\n"
            "  \"key\": \"C\",  // tonic note (C, D, E, F, G, A, B, with # or b)\n"
            "  \"mode\": \"major\",  // major or minor\n"
            "  \"time_signature\": \"4/4\",\n"
            "  \"tempo_bpm\": 100,  // realistic tempo for the song\n"
            "  \"measures\": [[\"1\",\"2\",\"3\",\"4\"],[\"5\",\"6\",\"7\",\"1\"]],  // organized by measure\n"
            "  \"numbers\": [\"1\",\"2\",\"3\",\"4\",\"5\",...],  // flattened from measures\n"
            "  \"lyrics\": \"First line of song\",\n"
            "  \"notes\": \"Additional context\"\n"
            "}\n\n"
            f"Song to transcribe: {query}\n\n"
            "Remember: Return the ACTUAL melody that people would recognize, not random sequences!"
        )
        try:
            resp = self.model.generate_content(prompt)
            if not resp or not resp.text:
                logger.error("Gemini song lookup returned empty response")
                return {
                    "found": False,
                    "numbers": [],
                    "lyrics": "",
                    "notes": "Gemini empty response",
                    "error": "gemini_failed",
                }
            logger.info("Gemini raw song response: %r", resp.text)
            text = resp.text.strip().strip("`")
            if text.startswith("json"):
                text = text[text.find("{") :]
            try:
                data = json.loads(text)
            except Exception as exc:
                logger.error("Gemini parse error: %s; raw=%r", exc, resp.text)
                return {
                    "found": False,
                    "numbers": [],
                    "lyrics": "",
                    "notes": f"Gemini parse error: {exc}",
                    "error": "gemini_failed",
                }
            numbers = data.get("numbers") or []
            measures = data.get("measures") or []
            if not numbers and measures:
                # flatten measures to numbers
                numbers = [tok for bar in measures for tok in bar]
            return {
                "found": bool(data.get("found")),
                "key": data.get("key"),
                "mode": data.get("mode"),
                "time_signature": data.get("time_signature"),
                "tempo_bpm": data.get("tempo_bpm"),
                "measures": measures,
                "numbers": numbers,
                "lyrics": data.get("lyrics", ""),
                "notes": data.get("notes", ""),
            }
        except Exception as exc:  # pragma: no cover
            logger.error("Gemini song lookup failed: %s", exc)
            return {
                "found": False,
                "numbers": [],
                "lyrics": "",
                "notes": f"Gemini error: {exc}",
                "error": "gemini_failed",
            }

    def _fallback(self, base: str, accuracy: float) -> str:
        if accuracy >= 90:
            return "Nice work! You're very close—keep the airflow steady and repeat once more for consistency."
        if accuracy >= 70:
            return "Good effort. Watch the intonation on the missed notes and keep the tempo steady."
        return "Let's try again slower. Focus on matching each pitch; breathe evenly and aim for clean note starts."
