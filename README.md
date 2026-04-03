# HumMuse
A modular, explainable multimodal co-creative architecture bridging acoustic input, symbolic structure, and harmonic reasoning.

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
- `POST /session/create`
- `GET /session/{id}/state`
- `GET /sessions`
- `POST /melody/from-hum` (multipart: `audio`, optional `prompt`, `mood`, `tempo_bpm`)
- `POST /chords/from-melody`
- `POST /chords/from-lyrics`
- `POST /melody/continue`
- `POST /suggest/lyrics`
- `PATCH /session/{id}/refine`
- `POST /session/{id}/chat`
- `POST /writerblock/help`
- `GET /artifact/{artifact_id}`
- `GET /artifact/{artifact_id}/{filename}`

EPIC 2 smoke check:
```bash
python -m evaluation.smoke_epic02
```

## UI
Run the Streamlit shell:
```bash
streamlit run ui/app.py
```

Current UI features:
- Sidebar session management: create new or load existing sessions
- Main workspace tabs: `Melody`, `Chords`, `Suggestions`, `Explanations`
- Active session id and current session summary displayed in the app
- Melody tab audio input via recorder or file upload, wired to `POST /melody/from-hum`
- Returned note events displayed with confidence color bands
- Chords tab supports both melody-based recognition and lyric-based generation
- Recognised chords show expandable mock decoding traces
- Returned chord progressions displayed as formatted cards with harmonic function, scores, and explanations
- Suggestions tab wired to `POST /melody/continue` and `POST /suggest/lyrics`
- Melody and lyric suggestions displayed as numbered options
- Refinement panel at the bottom of each tab, wired to `PATCH /session/{id}/refine`
- Latest refinement hint and interpretation string displayed in the session overview after refresh
- Explanations tab shows the current structured explanation report and a grounded chat panel wired to `POST /session/{id}/chat`
- Melody tab playback button that renders session melody material to WAV and plays it in the UI
- Optional SoundFont path for `midi2audio`; falls back to built-in note synthesis when MIDI rendering dependencies are unavailable
