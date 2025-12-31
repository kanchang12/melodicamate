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


def align_lyrics_with_numbers(lyrics_lines: List[str]) -> List[Dict[str, str]]:
    """
    Align lyric words to each number token based on alternating lyric/number lines
    returned by Gemini. This is a best-effort map so coaching can mention words.
    """
    aligned: List[Dict[str, str]] = []
    if not lyrics_lines:
        return aligned
    for i in range(0, len(lyrics_lines), 2):
        lyric_line = str(lyrics_lines[i] or "").strip()
        number_line = str(lyrics_lines[i + 1] if i + 1 < len(lyrics_lines) else "").strip()
        words = [w for w in lyric_line.split() if w]
        numbers = [n for n in number_line.replace(".", " ").split() if n]
        for idx, num in enumerate(numbers):
            word = words[idx] if idx < len(words) else (words[-1] if words else "")
            aligned.append({"lyric": word, "number": num})
    return aligned


def describe_mistakes_with_lyrics(
    wrong_notes: List[Dict],
    expected: List[str],
    played: List[str],
    key_tonic: str,
    mode: str,
    lyric_alignment: List[Dict[str, str]],
) -> List[str]:
    details: List[str] = []
    for wrong in wrong_notes:
        idx = wrong["index"]
        exp_num = expected[idx] if idx < len(expected) else wrong.get("expected")
        got_num = played[idx] if idx < len(played) else wrong.get("got")
        exp_note = pd_library.number_to_note_name(str(exp_num), key_tonic, mode) if exp_num else ""
        got_note = pd_library.number_to_note_name(str(got_num), key_tonic, mode) if got_num else ""
        lyric = lyric_alignment[idx].get("lyric", "") if idx < len(lyric_alignment) else ""
        phrase = f"Note {idx + 1}"
        if lyric:
            phrase += f" ('{lyric}')"
        if exp_num:
            phrase += f" expected {exp_num} ({exp_note})"
        if got_num:
            phrase += f" but sang {got_num} ({got_note})"
        else:
            phrase += " but was missing"
        details.append(phrase)
    return details


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


    @app.route("/api/song/analyze-recording", methods=["POST"])
    @rate_limited
    def analyze_recording():
        """
        Analyze a user's audio recording to extract melody as scale degrees.
        User records themselves singing/humming, we analyze it.
        
        Expects:
        - audio file in request.files['audio']
        - song_title (optional) in request.form
        """
        if 'audio' not in request.files:
            return (
                jsonify({
                    "error": "invalid_input",
                    "message": "No audio file provided. Please upload an audio file."
                }),
                400,
            )
        
        audio_file = request.files['audio']
        song_title = request.form.get('song_title', 'Your Recording')
        
        # Read audio data
        audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return (
                jsonify({
                    "error": "invalid_input",
                    "message": "Audio file is empty"
                }),
                400,
            )
        
        # Analyze with Gemini
        result = gemini.analyze_user_recording(audio_data, song_title)
        
        if not result.get("found"):
            error = result.get("error", "unknown")
            notes = result.get("notes", "")
            
            return (
                jsonify({
                    "found": False,
                    "error": error,
                    "message": "Could not extract melody from recording. Try recording again with clearer audio.",
                    "notes": notes
                }),
                200,  # Not a server error, just couldn't detect melody
            )
        
        # Success - melody detected
        numbers = result.get("numbers", [])
        key = result.get("key", "C")
        mode = result.get("mode", "major")
        
        # Convert scale degrees to note names
        note_names = []
        for token in numbers:
            try:
                from services import pd_library
                note_name = pd_library.number_to_note_name(token, key, mode)
                note_names.append(note_name)
            except:
                note_names.append(str(token))
        
        # Create a song ID from the title
        song_id = song_title.lower().replace(" ", "_").replace("'", "")
        
        return jsonify({
            "found": True,
            "song_id": song_id,
            "canonical_title": song_title,
            "title": song_title,
            "key": key,
            "mode": mode,
            "numbers": numbers,
            "note_names": note_names,
            "tempo_bpm": result.get("tempo_bpm", 100),
            "confidence": result.get("confidence", "unknown"),
            "notes": result.get("notes", ""),
            "source": "user_recording",
            "message": f"Successfully analyzed your recording! Detected {len(numbers)} notes."
        })

    @app.route("/api/coach/exercise", methods=["POST", "GET"])
    @rate_limited
    def coach_exercise():
        if request.method == "GET":
            return jsonify({"message": "POST exercise data for coaching."})

        data = request.get_json(force=True, silent=True) or {}
        
        # 1. Get the song metadata from the Flutter request
        exercise_id = data.get("exercise_id")
        notes = data.get("notes", [])
        key_tonic = data.get("key_tonic", "C")
        mode = data.get("mode", "major")
        
        # PRIORITY ORDER for getting the song title:
        # 1. canonical_title (from song request API)
        # 2. title (direct from Flutter)
        # 3. exercise_id (fallback to ID)
        # 4. "this piece" (last resort)
        canonical_title = (
            data.get("canonical_title") or 
            data.get("title") or 
            data.get("song_title") or
            exercise_id or 
            "this piece"
        )
        
        # 2. Get the target melody
        expected_numbers = [str(x) for x in data.get("expected_numbers") or []]

        if not expected_numbers:
            if exercise_id in EXPECTED_EXERCISES:
                expected_numbers = expected_numbers_for_exercise(exercise_id, key_tonic, mode)
            else:
                song_match = pd_library.find_song(exercise_id, {})
                expected_numbers = song_match["numbers"] if song_match else []

        # 3. Map the user's LIVE notes to numbers
        played_numbers = map_notes_to_numbers(notes, key_tonic, mode, "sharps")

        lyrics_lines = data.get("lyrics_lines") or []
        lyric_alignment = align_lyrics_with_numbers(lyrics_lines)

        if not expected_numbers:
            accuracy = 0.0
            mistake_summary = {"issues": [], "summary": "Free play session - no target melody found."}
            lyric_context = []
        else:
            # 4. Compare the sequences
            accuracy, wrong_notes = compare_sequences(expected_numbers, played_numbers)
            mistake_summary = build_mistake_summary(wrong_notes)
            
            # Create simple, lyric-focused context (not technical)
            lyric_context = []
            for wrong in wrong_notes[:3]:  # Only first 3 mistakes
                idx = wrong["index"]
                lyric = lyric_alignment[idx].get("lyric", "") if idx < len(lyric_alignment) else ""
                if lyric:
                    lyric_context.append(lyric)

        # 5. Create a simplified, musical prompt payload
        # Remove all technical details - just focus on song and performance
        prompt_payload = {
            "title": canonical_title,  # Use the proper title here!
            "accuracy_pct": accuracy,
            "lyric_context": lyric_context,  # Just the words, no note names
            "lyrics_lines": lyrics_lines,  # For additional context if needed
        }
        
        coaching_text = gemini.generate_coaching_text(prompt_payload)

        # 6. Generate TTS
        tts_audio = None
        voice_enabled = data.get("voice_enabled", True)
        if voice_enabled and coaching_text:
            tts_audio = eleven.text_to_speech(coaching_text, data.get("voice_id"))

        # Return both human coaching and technical details
        # (technical details for UI display, not for voice)
        return jsonify({
            "refused": False,
            "coaching_text": coaching_text,
            "song_title": canonical_title,  # Echo back the title for confirmation
            "expected_numbers": expected_numbers,
            "played_numbers": played_numbers,
            "mistakes_summary": mistake_summary,
            "accuracy": accuracy,
            "tts_audio_base64": tts_audio,
        })

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

        # Get classification info (not used for gating, just metadata)
        classification = gemini.classify_song_request(query, data.get("composer_or_artist"))
        
        # Use the user's original query as the canonical title
        # Only use classification title if user query is very short/unclear
        canonical_title = query if len(query) > 2 else classification.get("canonical_title", query)
        
        # Get song data from Gemini
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
                            "canonical_title": canonical_title,  # Return the user's query
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
                        "canonical_title": canonical_title,  # Return the user's query
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
        lines = song_data.get("lines", [])
        
        note_names = [pd_library.number_to_note_name(token, key, mode) for token in numbers]
        
        # Create a proper song ID from the title
        # This helps Flutter store it with a recognizable name
        song_id = canonical_title.lower().replace(" ", "_").replace("'", "")
        
        return jsonify(
            {
                "found": True,
                "refused": False,
                "song_id": song_id,  # e.g., "happy_birthday", "twinkle_twinkle"
                "canonical_title": canonical_title,  # User's original input
                "title": canonical_title,  # Alias for convenience
                "key": key,
                "mode": mode,
                "numbers": numbers,
                "measures": measures,
                "lines": lines,
                "tempo_bpm": tempo,
                "note_names": note_names,
                "lyrics": lyrics,
                "source": "gemini",
                "confidence": song_data.get("confidence", "unknown"),
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
