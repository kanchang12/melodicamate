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
        Use Gemini to return the full primary melody (verse + chorus) as scale-degree numbers (relative to tonic) and lyrics, interleaved line by line.
        Expected JSON (do not wrap in code fences):
        {
          "found": true/false,
          "key": "G",
          "mode": "major",
          "time_signature": "4/4",
          "tempo_bpm": 90,
          "lines": [
            "Twinkle twinkle little star",
            "1 1 5 5 6 6 5",
            "How I wonder what you are",
            "5 4 3 2 1"
          ],
          "notes": "any notes"
        }
        """
        if not self._enabled():
            notes = self.config_error or "Gemini disabled"
            return {
                "found": False,
                "lines": [],
                "notes": notes,
                "error": "gemini_failed",
            }
        prompt = (
            "You are a professional music transcription assistant. Given a song title (and optional artist), "
            "return the ACTUAL primary melody from the real song as scale-degree numbers (solfege) relative to the tonic, interleaved with lyrics.\n\n"
            "IMPORTANT RULES:\n"
            "1. Return the REAL, RECOGNIZABLE melody from the actual song - NOT random notes\n"
            "2. Include verse AND chorus (aim for 32-96 notes total with repeats)\n"
            "3. Alternate lines: first line is lyrics, second line is numbers, third line is lyrics, fourth line is numbers, etc.\n"
            "4. Use appropriate tempo for the song (e.g., 60-80 bpm for ballads, 100-140 for upbeat)\n"
            "5. Identify the correct key and mode (major/minor) for the song\n"
            "6. For rhythm: use single notes per beat OR break beats into smaller durations\n"
            "7. If lyrics are unknown, use '-' for the lyrics line.\n\n"
            "Response format (JSON only, no code fences):\n"
            "{\n"
            "  \"found\": true/false,\n"
            "  \"key\": \"C\",\n"
            "  \"mode\": \"major\",\n"
            "  \"time_signature\": \"4/4\",\n"
            "  \"tempo_bpm\": 100,\n"
            "  \"lines\": [\n"
            "    \"lyrics line 1\",\n"
            "    \"numbers line 1\",\n"
            "    \"lyrics line 2\",\n"
            "    \"numbers line 2\",\n"
            "    ...\n"
            "  ],\n"
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
                    "lines": [],
                    "notes": f"Gemini parse error: {exc}",
                    "error": "gemini_failed",
                }
            return {
                "found": bool(data.get("found")),
                "key": data.get("key"),
                "mode": data.get("mode"),
                "time_signature": data.get("time_signature"),
                "tempo_bpm": data.get("tempo_bpm"),
                "lines": data.get("lines", []),
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
