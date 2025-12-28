# MelodicaMate Backend

Flask service that powers the MelodicaMate voice-first tutor. Runs locally or on Cloud Run.

## Setup
1. `python -m venv .venv && .\\.venv\\Scripts\\activate` (PowerShell)  
2. `pip install -r requirements.txt`
3. `copy .env.example .env` and set keys: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `APP_TTS_CHAR_LIMIT`.

## Run (local)
```bash
flask --app app run --debug
```

## Tests
```bash
python -m pytest
```

## Docker & Cloud Run
```bash
docker build -t melodicamate-backend .
docker run -p 8080:8080 --env-file .env melodicamate-backend
gcloud builds submit --tag gcr.io/PROJECT_ID/melodicamate-backend
gcloud run deploy melodicamate-backend --image gcr.io/PROJECT_ID/melodicamate-backend --platform managed --region REGION --allow-unauthenticated --min-instances=0 --max-instances=1
```

## Key Endpoints
- `GET /health`
- `POST /api/coach/exercise`
- `POST /api/transcribe/notes-to-numbers`
- `POST /api/song/request`
- `POST /api/tts`

## Notes
- Gemini usage is stubbed offline; set `GEMINI_API_KEY` to enable remote calls.
- ElevenLabs TTS returns `None` if not configured; the app handles missing audio gracefully.
