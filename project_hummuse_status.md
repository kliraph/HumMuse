---
name: HumMuse project status and plan
description: Current implementation state vs Implementation_Plan_v3_3.docx
type: project
---

HumMuse is a Masters thesis project: an AI co-creative songwriting assistant with melody extraction from hum, chord support, melodic continuation, lyric suggestions, and explainable interaction.

**Plan documents in repo root:**
- `Implementation_Plan_v3_3.docx` - ~60 tasks, 9 phases, about 252 hours
- `AI_CoCreative_System_Architecture_v3.1.docx` - updated v3.1 architecture and explainability design

**Naming divergences (plan vs actual code):**
- Plan `/api` maps to repo `backend/`
- Plan `/ml/melody`, `/ml/chords`, `/ml/continuation`, `/ml/lyrics`, `/ml/recognition` map to repo modules such as `ml/melody_sketchpad`, `ml/lyrics_to_chords`, and `ml/writers_block`
- Plan `/eval` maps to `evaluation/`
- Plan `/tests` does not exist as a top-level folder; tests currently live in `evaluation/`

**Complete and aligned to v3.3 through Phase 4 as DQN with reward-attributed XAI as of 2026-05-02:**
- Phase 0 scaffold is present: repo layout, README, adapted directories, partial Makefile, per-module requirements
- Phase 1 Tasks 1.1-1.5 are updated to the v3.3 scaffold:
  - `shared/schemas.py` now includes DQN chord distributions, chord explanations, chord-symbol annotations, `chat_history`, `explanation_report`, `RefinementPlan`, and `ChatMessage`
  - `backend/database.py` now includes `sessions`, `api_cache` with `use_case`, and `metrics`; BACHI `decoding_traces` storage is removed
  - `backend/session.py` preserves the v3.1 session fields
  - request logging includes `use_case` for GPT-style endpoints
  - stub API includes `POST /session/{id}/chat`; the BACHI `POST /chords/from-melody` endpoint is removed
- Phase 2 Tasks 2.1-2.6 are updated to the v3.3 UI scaffold:
  - Streamlit app now has four tabs: `Melody`, `Chords`, `Suggestions`, `Explanations`
  - Melody notes render with confidence color bands
  - Chords tab now supports the single lyric-driven DQN generation pathway with expandable native Q-values, q-margins, reward attribution, and chord explanations
  - Explanations tab shows the current `ExplanationReport` plus grounded chat history/input, without BACHI decoding traces
  - refinement responses now expose an interpretation string in the UI/session overview
  - MIDI playback remains available with `midi2audio` or note-synthesis fallback
- Validation is green:
  - `python -m pytest evaluation/test_session.py evaluation/test_api.py evaluation/test_ui.py`
  - `python -m evaluation.smoke_epic02`

**Still stubbed / not yet implemented:**
- Real melody extraction pipeline (Phase 3)
- Full DistilBERT/GoEmotions emotion model loading is still represented by lightweight deterministic mapping
- DQN fine-tuning/stretch evaluation beyond the vendored Wiki-DQN-64 checkpoint
- Real melody continuation pipeline (Phase 5)
- Unified GPT client, cache, prompt templates, and routing (Phase 6)
- Refinement execution and explainability aggregation beyond the current stub layer (Phase 7)
- Golden set, metrics, and broader evaluation framework (Phase 8)
- `docker-compose.yml` and fuller Makefile targets from Phase 0 remain unfinished

**Why this matters:**
The repo now matches the updated v3.3 single-path chord architecture through the Phase 4 DQN API/UI/schema surface with reward-attributed XAI. The remaining work is real GPT integration, melody continuation, evaluation hardening, and optional DQN ablation work.
