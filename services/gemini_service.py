import json
import os
from typing import Dict

import logging
import google.generativeai as genai

logger = logging.getLogger("melodicamate.gemini")


class GeminiService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-pro")
        else:
            self.model = None

    def _enabled(self) -> bool:
        return bool(self.api_key and self.model)

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
            return {"found": False, "numbers": [], "lyrics": "", "notes": "Gemini disabled"}
        prompt = (
            "You are a music assistant. Given a song title (and optional artist), return its main melody "
            "as scale-degree numbers in a likely key, plus one-line lyrics snippet if known. "
            "Respond with JSON only, fields: found (bool), numbers (array of strings), lyrics (string), notes (string). "
            f"Song: {query}"
        )
        try:
            resp = self.model.generate_content(prompt)
            if not resp or not resp.text:
                return {"found": False, "numbers": [], "lyrics": "", "notes": "No response"}
            text = resp.text.strip().strip("`")
            if text.startswith("json"):
                text = text[text.find("{") :]
            data = json.loads(text)
            return {
                "found": bool(data.get("found")),
                "numbers": data.get("numbers", []),
                "lyrics": data.get("lyrics", ""),
                "notes": data.get("notes", ""),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini song lookup failed: %s", exc)
            return {"found": False, "numbers": [], "lyrics": "", "notes": "Gemini error"}

    def _fallback(self, base: str, accuracy: float) -> str:
        if accuracy >= 90:
            return "Nice work! You're very close—keep the airflow steady and repeat once more for consistency."
        if accuracy >= 70:
            return "Good effort. Watch the intonation on the missed notes and keep the tempo steady."
        return "Let's try again slower. Focus on matching each pitch; breathe evenly and aim for clean note starts."
