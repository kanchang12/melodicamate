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
        expected = payload.get("expected") or []
        played = payload.get("played") or []
        expected_notes = payload.get("expected_note_names") or []
        played_notes = payload.get("played_note_names") or []
        lyric_context = payload.get("lyric_context") or []
        lyrics_lines = payload.get("lyrics_lines") or []
        title = payload.get("title") or payload.get("exercise_id")
        base = (
            f"{title} in key {payload.get('key')} {payload.get('mode')} "
            f"accuracy {accuracy:.1f}%. {mistakes}"
        )
        if not self._enabled():
            return self._fallback(base, accuracy)
        try:
            prompt_parts = [
                "You are a concise music coach. Keep replies to <=2 sentences, max 320 chars.",
                "Reference the lyric word near each mistake when provided.",
                f"Song: {title}. Key: {payload.get('key')} {payload.get('mode')}.",
                f"Accuracy: {accuracy:.1f}%. Mistakes summary: {mistakes}.",
                f"Expected numbers/notes: {' '.join(expected)} | {' '.join(expected_notes)}.",
                f"Played numbers/notes: {' '.join(played)} | {' '.join(played_notes)}.",
            ]
            if lyric_context:
                prompt_parts.append("Lyric positions: " + " | ".join(lyric_context[:4]) + ".")
            if lyrics_lines:
                prompt_parts.append("Reference lyric/number lines:\n" + "\n".join(lyrics_lines[:6]))
            prompt = " ".join(prompt_parts)
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
        Use Gemini to return interleaved lyrics and scale-degree numbers.
        Returns a single formatted string in 'lyrics' for the Flutter UI.
        """
        if not self._enabled():
            return {"found": False, "notes": "Gemini disabled", "error": "gemini_failed"}

        prompt = (
            "You are a professional music transcription assistant. Given a song title, "
            "return the melody as scale-degree numbers relative to the tonic, interleaved with lyrics.\n\n"
            "RULES:\n"
            "1. Alternate lines: Line 1=Lyrics, Line 2=Numbers, Line 3=Lyrics, Line 4=Numbers.\n"
            "2. Numbers should be the actual melody notes.\n"
            "3. Return ONLY valid JSON.\n\n"
            "Response format:\n"
            "{\n"
            '  "found": true,\n'
            '  "key": "G",\n'
            '  "mode": "major",\n'
            '  "time_signature": "4/4",\n'
            '  "tempo_bpm": 90,\n'
            '  "lines": ["Lyric line", "1 1 5 5", "Lyric line", "6 6 5"],\n'
            '  "notes": "context"\n'
            "}\n\n"
            f"Song: {query}"
        )

        try:
            resp = self.model.generate_content(prompt)
            if not resp or not resp.text:
                return {"found": False, "error": "empty_response"}

            text = resp.text.strip().strip("`").replace("json", "", 1).strip()
            data = json.loads(text)

            raw_lines = data.get("lines", [])
            numbers_only = []

            # --- INTERLEAVING LOGIC FOR FLUTTER ---
            # We join all lines with newlines so it looks like a lead sheet
            # Line 1 (Lyric)
            # Line 2 (Numbers)
            interleaved_string = "\n".join(raw_lines)

            # Extract just the numbers for the app's player/engine
            for i, line in enumerate(raw_lines):
                if i % 2 != 0:  # Odd indices (1, 3, 5) are the number lines
                    cleaned = line.replace(".", " ")
                    numbers_only.extend(cleaned.split())

            return {
                "found": bool(data.get("found")),
                "key": data.get("key"),
                "mode": data.get("mode"),
                "time_signature": data.get("time_signature"),
                "tempo_bpm": data.get("tempo_bpm"),
                "lines": raw_lines,
                "numbers": numbers_only,  # Used for the playback logic
                "lyrics": interleaved_string,  # Used for the UI display
                "notes": data.get("notes", ""),
            }
        except Exception as exc:
            logger.error("Gemini song lookup failed: %s", exc)
            return {"found": False, "error": str(exc)}

    def _fallback(self, base: str, accuracy: float) -> str:
        if accuracy >= 90:
            return "Nice work! You're very close—keep the airflow steady and repeat once more for consistency."
        if accuracy >= 70:
            return "Good effort. Watch the intonation on the missed notes and keep the tempo steady."
        return "Let's try again slower. Focus on matching each pitch; breathe evenly and aim for clean note starts."
