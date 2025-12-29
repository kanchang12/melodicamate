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
        Use Gemini to return the full primary melody and lyrics.
        Now correctly extracts lyrics from the interleaved lines.
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
            "You are a professional music transcription assistant. Given a song title, "
            "return the ACTUAL primary melody as scale-degree numbers relative to the tonic, interleaved with lyrics.\n\n"
            "IMPORTANT RULES:\n"
            "1. Return the REAL, RECOGNIZABLE melody.\n"
            "2. Alternate lines: first line is lyrics, second line is numbers.\n"
            "3. Use appropriate tempo and identify correct key/mode.\n\n"
            "Response format (JSON only, no code fences):\n"
            "{\n"
            "  \"found\": true,\n"
            "  \"key\": \"C\",\n"
            "  \"mode\": \"major\",\n"
            "  \"time_signature\": \"4/4\",\n"
            "  \"tempo_bpm\": 100,\n"
            "  \"lines\": [\"lyrics 1\", \"1 2 3\", \"lyrics 2\", \"3 2 1\"],\n"
            "  \"notes\": \"context\"\n"
            "}\n\n"
            f"Song: {query}"
        )

        try:
            resp = self.model.generate_content(prompt)
            if not resp or not resp.text:
                return {"found": False, "error": "gemini_failed"}

            # Clean the JSON response
            text = resp.text.strip().strip("`")
            if text.startswith("json"):
                text = text[text.find("{") :]
            
            data = json.loads(text)
            lines = data.get("lines") or []
            
            # --- FIX: EXTRACTION LOGIC ---
            numbers = []
            lyric_parts = []
            
            for i, line in enumerate(lines):
                if not isinstance(line, str):
                    continue
                
                # Even indices (0, 2, 4...) are Lyrics
                if i % 2 == 0:
                    lyric_parts.append(line)
                # Odd indices (1, 3, 5...) are Numbers
                else:
                    cleaned = line.replace(".", " ")
                    numbers.extend(cleaned.split())
            
            # Join the lyrics list into a single string with newlines for Flutter
            full_lyrics = "\n".join(lyric_parts)
            # -----------------------------

            return {
                "found": bool(data.get("found")),
                "key": data.get("key"),
                "mode": data.get("mode"),
                "time_signature": data.get("time_signature"),
                "tempo_bpm": data.get("tempo_bpm"),
                "lines": lines,
                "numbers": numbers,
                "lyrics": full_lyrics,  # This now contains the actual text!
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
