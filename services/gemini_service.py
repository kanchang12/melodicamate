import json
import os
from typing import Dict

import logging

logger = logging.getLogger("melodicamate.gemini")


class GeminiService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")

    def _enabled(self) -> bool:
        return bool(self.api_key)

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
            # Placeholder for real Gemini call; offline environments use fallback.
            return self._fallback(base, accuracy)
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini fallback due to error: %s", exc)
            return self._fallback(base, accuracy)

    def classify_song_request(self, query: str, composer: str = "") -> Dict:
        likely_pd = any(
            name.lower() in query.lower() for name in ["beethoven", "mozart", "bach", "ode to joy", "twinkle"]
        )
        return {
            "likely_public_domain": likely_pd,
            "canonical_title": query.title(),
            "composer": composer or "",
            "search_terms": [query],
            "notes": "Heuristic classification (offline).",
        }

    def _fallback(self, base: str, accuracy: float) -> str:
        if accuracy >= 90:
            return "Nice work! You're very close—keep the airflow steady and repeat once more for consistency."
        if accuracy >= 70:
            return "Good effort. Watch the intonation on the missed notes and keep the tempo steady."
        return "Let's try again slower. Focus on matching each pitch; breathe evenly and aim for clean note starts."
