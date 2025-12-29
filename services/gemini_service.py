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
        Use Gemini to return a melody as scale-degree numbers (relative to tonic) and lyrics if known.
        Expected JSON:
        {"found":true/false,"numbers":["1","2",...],"lyrics":"...","notes":"..."}
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
            "You are a music assistant. Given a song title (and optional artist), return its main melody "
            "as scale-degree numbers in a likely key, plus one-line lyrics snippet if known. "
            "Respond with JSON only, fields: found (bool), numbers (array of strings), lyrics (string), notes (string). "
            f"Song: {query}"
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
            return {
                "found": bool(data.get("found")),
                "numbers": data.get("numbers", []),
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
