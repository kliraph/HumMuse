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

## Model Assets
Vendored Melody Transformer continuation artifacts live under
`storage/models/Lakh-MT/`.

Vendored DQN harmony artifacts live under `storage/models/Wiki-DQN-64/`.

| File | SHA-256 |
| --- | --- |
| `best_data_augmented.pt` | `d5ee90bd14e5420e15417e3280b529bfb76f0ea05a6541fd62eb356562c0a17e` |
| `tokenizer.json` | `5117e7aeebd5cc43365082e461f0401b0906bf37d4ad40dbc2815c3525cb2a96` |
| `Wiki-DQN-64/epoch14_reward4.298_mle_loss262.858_beta0.700.pth` | `f968488e169e8e398fa0c29cd66fd542a8fb29a5b10a9876003163746423fb86` |

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
- Chords tab supports lyric-based DQN chord generation with reward-attributed XAI
- Returned chord progressions displayed as formatted cards with harmonic function, scores, annotations, native DQN Q-values, margins, and explanations
- Suggestions tab wired to `POST /melody/continue` and `POST /suggest/lyrics`
- Melody and lyric suggestions displayed as numbered options
- Refinement panel at the bottom of each tab, wired to `PATCH /session/{id}/refine`
- Latest refinement hint and interpretation string displayed in the session overview after refresh
- Explanations tab shows the current structured explanation report and a grounded chat panel wired to `POST /session/{id}/chat`
- Melody tab playback button that renders session melody material to WAV and plays it in the UI
- Optional SoundFont path for `midi2audio`; falls back to built-in note synthesis when MIDI rendering dependencies are unavailable
