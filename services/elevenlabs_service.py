import base64
import os
from typing import Optional

import logging
import requests

logger = logging.getLogger("melodicamate.eleven")


class ElevenLabsService:
    def __init__(self, char_limit: int = 400) -> None:
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.default_voice = os.getenv("ELEVENLABS_VOICE_ID", "")
        self.char_limit = char_limit

    def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[str]:
        if not text:
            return None
        text = text[: self.char_limit]
        if not self.api_key:
            logger.info("ElevenLabs key missing; returning no audio.")
            return None
        voice = voice_id or self.default_voice
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
            headers = {"xi-api-key": self.api_key, "accept": "audio/mpeg"}
            resp = requests.post(
                url,
                headers=headers,
                json={"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5}},
                timeout=15,
            )
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("ElevenLabs TTS failed: %s", exc)
            return None
