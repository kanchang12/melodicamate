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
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using ElevenLabs API.
        Returns base64-encoded audio data or None if failed.
        """
        if not text:
            logger.debug("No text provided for TTS")
            return None
        
        # Truncate text if too long
        if len(text) > self.char_limit:
            text = text[:self.char_limit]
            logger.info(f"Text truncated to {self.char_limit} characters")
        
        if not self.api_key:
            logger.warning("ElevenLabs API key missing; TTS unavailable")
            return None
        
        voice = voice_id or self.default_voice
        if not voice:
            logger.error("No voice ID provided and no default voice configured")
            return None
        
        try:
            url = f"{self.base_url}/text-to-speech/{voice}"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "accept": "audio/mpeg"
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",  # Updated to latest model
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            logger.info(f"Sending TTS request: {len(text)} chars, voice: {voice[:8]}...")
            
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=15,
            )
            
            # Check for errors
            if resp.status_code == 401:
                logger.error("ElevenLabs authentication failed - check API key")
                return None
            elif resp.status_code == 400:
                logger.error(f"Bad request to ElevenLabs: {resp.text}")
                return None
            elif resp.status_code == 429:
                logger.warning("ElevenLabs rate limit exceeded")
                return None
            
            resp.raise_for_status()
            
            # Verify we got audio data
            if len(resp.content) == 0:
                logger.error("ElevenLabs returned empty audio")
                return None
            
            audio_base64 = base64.b64encode(resp.content).decode("utf-8")
            logger.info(f"TTS successful: {len(resp.content)} bytes encoded")
            
            return audio_base64
            
        except requests.exceptions.Timeout:
            logger.warning("ElevenLabs TTS request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to ElevenLabs API")
            return None
        except Exception as exc:
            logger.error(f"ElevenLabs TTS failed: {type(exc).__name__}: {exc}")
            return None
    
    def get_available_voices(self) -> Optional[list]:
        """
        Fetch available voices from ElevenLabs API.
        Useful for letting users choose their preferred voice.
        """
        if not self.api_key:
            return None
        
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            voices = data.get("voices", [])
            
            logger.info(f"Retrieved {len(voices)} available voices")
            return voices
            
        except Exception as exc:
            logger.error(f"Failed to fetch voices: {exc}")
            return None
