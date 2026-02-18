# HumMuse
A multimodal AI assistant for songwriting that transforms hummed melodies and lyrical input into structured musical suggestions

## Project scaffold
- `backend/` - backend services, storage and logging
- `ml/` - ML modules
- `ui/` - UI modules
- `evaluation/` - evaluation scripts/checks
- `shared/` - shared schemas
- `storage/` - runtime storage for artifacts and SQLite DB

## Setup
```bash
pip install -e .
```

## EPIC 0-1 smoke check
```bash
python -m evaluation.smoke_epic01
```

## EPIC 2 API
Run API server:
```bash
uvicorn backend.app:app --reload
```

Endpoints:
- `GET /health`
- `GET /version`
- `POST /melody/from-hum` (multipart: `audio`, optional `prompt`, `mood`, `tempo_bpm`)
- `POST /chords/from-lyrics`
- `POST /writerblock/help`
- `GET /artifact/{artifact_id}/{filename}`

EPIC 2 smoke check:
```bash
python -m evaluation.smoke_epic02
```
