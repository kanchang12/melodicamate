import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from services.music_mapping import (
    degree_token_for_midi,
    map_notes_to_numbers,
    midi_to_note_name,
)
from services.scoring import (
    EXPECTED_EXERCISES,
    build_mistake_summary,
    compare_sequences,
    expected_numbers_for_exercise,
)
from services import pd_library
from services.gemini_service import GeminiService
from services.elevenlabs_service import ElevenLabsService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("melodicamate")

APP_TTS_CHAR_LIMIT = int(os.getenv("APP_TTS_CHAR_LIMIT", "400"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))


def create_app() -> Flask:
    app = Flask(__name__)

    rate_state: Dict[str, Dict[str, Any]] = {}

    def rate_limited(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = request.remote_addr or "unknown"
            now = time.time()
            bucket = rate_state.get(ip, {"count": 0, "reset": now + 60})
            if now > bucket["reset"]:
                bucket = {"count": 0, "reset": now + 60}
            if bucket["count"] >= RATE_LIMIT_PER_MIN:
                return (
                    jsonify(
                        {
                            "error": "rate_limited",
                            "message": "Please slow down; try again in a minute.",
                        }
                    ),
                    429,
                )
            bucket["count"] += 1
            rate_state[ip] = bucket
            return fn(*args, **kwargs)

        return wrapper

    gemini = GeminiService()
    eleven = ElevenLabsService(char_limit=APP_TTS_CHAR_LIMIT)

    @app.errorhandler(400)
    def handle_bad_request(err):
        return (
            jsonify({"error": "bad_request", "message": str(err)}),
            400,
        )

    @app.route("/health", methods=["GET", "POST"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/", methods=["GET", "POST"])
    def root():
        return jsonify(
            {
                "status": "ok",
                "message": "MelodicaMate backend is running",
                "routes": [
                    "/health",
                    "/api/coach/exercise",
                    "/api/transcribe/notes-to-numbers",
                    "/api/song/request",
                    "/api/tts",
                ],
            }
        )

    @app.route("/api/transcribe/notes-to-numbers", methods=["POST", "GET"])
    @rate_limited
    def notes_to_numbers():
        if request.method == "GET":
            return jsonify(
                {
                    "message": "POST notes to transcribe.",
                    "example": {
                        "key_tonic": "C",
                        "mode": "major",
                        "accidental_pref": "sharps",
                        "notes": [{"midi": 60, "t0_ms": 0, "t1_ms": 500, "conf": 0.9}],
                    },
                }
            )
        data = request.get_json(force=True, silent=True) or {}
        key_tonic = data.get("key_tonic", "C")
        mode = data.get("mode", "major")
        acc_pref = data.get("accidental_pref", "sharps")
        notes = data.get("notes", [])
        if not isinstance(notes, list):
            return (
                jsonify({"error": "invalid_input", "message": "notes must be a list"}),
                400,
            )
        numbers = map_notes_to_numbers(notes, key_tonic, mode, acc_pref)
        note_names = [midi_to_note_name(n.get("midi")) for n in notes]
        return jsonify({"numbers": numbers, "note_names": note_names})

    @app.route("/api/coach/exercise", methods=["POST", "GET"])
    @rate_limited
    def coach_exercise():
        if request.method == "GET":
            return jsonify(
                {
                    "message": "POST exercise data for coaching.",
                    "expected_fields": [
                        "exercise_id",
                        "key_tonic",
                        "mode",
                        "notation",
                        "notes",
                        "metrics",
                    ],
                }
            )
        data = request.get_json(force=True, silent=True) or {}
        metrics = data.get("metrics", {}) or {}
        if metrics.get("suspected_recording"):
            return (
                jsonify(
                    {
                        "refused": True,
                        "reason": "suspected_recording",
                        "message": "Sorry, this sounds like a recording. I can only analyze live singing or live playing.",
                    }
                ),
                403,
            )

        exercise_id = data.get("exercise_id")
        notes = data.get("notes", [])
        key_tonic = data.get("key_tonic", "C")
        mode = data.get("mode", "major")
        notation = data.get("notation", "numbers")

        played_numbers = map_notes_to_numbers(notes, key_tonic, mode, "sharps")
        if exercise_id not in EXPECTED_EXERCISES:
            # Allow free-play: no strict expectations; just echo back.
            expected_numbers = []
            accuracy = 0.0
            mistake_summary = {"issues": [], "summary": "Free play session."}
        else:
            expected_numbers = expected_numbers_for_exercise(exercise_id, key_tonic, mode)
            accuracy, wrong_notes = compare_sequences(expected_numbers, played_numbers)
            mistake_summary = build_mistake_summary(wrong_notes)

        prompt_payload = {
            "exercise_id": exercise_id,
            "key": key_tonic,
            "mode": mode,
            "accuracy_pct": accuracy,
            "mistakes": mistake_summary,
        }
        coaching_text = gemini.generate_coaching_text(prompt_payload)

        tts_audio = None
        voice_enabled = (data.get("voice_enabled", True),)[0]
        if voice_enabled and coaching_text:
            tts_audio = eleven.text_to_speech(coaching_text, data.get("voice_id"))

        return jsonify(
            {
                "refused": False,
                "coaching_text": coaching_text,
                "expected_numbers": expected_numbers,
                "played_numbers": played_numbers,
                "mistakes_summary": mistake_summary,
                "notation": notation,
                "tts_audio_base64": tts_audio,
            }
        )

    @app.route("/api/song/request", methods=["POST", "GET"])
    @rate_limited
    def song_request():
        if request.method == "GET":
            return jsonify({"message": "POST a song query to search public-domain library."})
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        desired_key = data.get("desired_key", "C")
        mode = data.get("mode", "major")
        if not query:
            return (
                jsonify({"error": "invalid_input", "message": "query is required"}),
                400,
            )

        classification = gemini.classify_song_request(query, data.get("composer_or_artist"))
        song_data = gemini.generate_song_numbers(query)
        if not song_data.get("found"):
            error = song_data.get("error")
            notes = song_data.get("notes", "")
            if error:
                logger.error("Gemini lookup failed: %s | notes=%s", error, notes)
                return (
                    jsonify(
                        {
                            "found": False,
                            "refused": False,
                            "error": error,
                            "message": "Gemini call failed or returned invalid data.",
                            "canonical": classification,
                            "notes": notes,
                        }
                    ),
                    502,
                )
            return (
                jsonify(
                    {
                        "found": False,
                        "refused": False,
                        "message": "No open version found. Please sing or play it live.",
                        "canonical": classification,
                        "notes": notes,
                    }
                ),
                200,
            )

        numbers = song_data.get("numbers", [])
        lyrics = song_data.get("lyrics", "")
        key = song_data.get("key") or desired_key
        mode = song_data.get("mode") or mode
        tempo = song_data.get("tempo_bpm")
        measures = song_data.get("measures", [])
        note_names = [pd_library.number_to_note_name(token, key, mode) for token in numbers]
        return jsonify(
            {
                "found": True,
                "refused": False,
                "canonical": classification,
                "key": key,
                "mode": mode,
                "numbers": numbers,
                "measures": measures,
                "tempo_bpm": tempo,
                "note_names": note_names,
                "lyrics": lyrics,
                "source": "gemini",
                "notes": song_data.get("notes", ""),
            }
        )

    @app.route("/api/tts", methods=["POST", "GET"])
    @rate_limited
    def tts():
        if request.method == "GET":
            return jsonify({"message": "POST text to synthesize.", "limit_chars": APP_TTS_CHAR_LIMIT})
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        voice_id = data.get("voice_id")
        if not text:
            return (
                jsonify({"error": "invalid_input", "message": "text is required"}),
                400,
            )
        if len(text) > APP_TTS_CHAR_LIMIT:
            text = text[:APP_TTS_CHAR_LIMIT]
        audio_base64 = eleven.text_to_speech(text, voice_id)
        if not audio_base64:
            return (
                jsonify(
                    {
                        "error": "tts_failed",
                        "message": "Text-to-speech unavailable. Check configuration.",
                    }
                ),
                500,
            )
        return jsonify({"audio_base64": audio_base64, "mime": "audio/mpeg"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
