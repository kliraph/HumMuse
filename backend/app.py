"""FastAPI app that orchestrates backend generation flows."""

from __future__ import annotations

import mimetypes
import os
import time
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from midiutil import MIDIFile

from backend.api_models import (
    ArtifactListingResponse,
    ChordsFromMelodyRequest,
    ChordsFromMelodyResponse,
    ChordsManualRequest,
    ChordsManualResponse,
    LyricsToChordsRequest,
    LyricsToChordsResponse,
    MelodyAcceptRequest,
    MelodyContinueRequest,
    MelodyContinueResponse,
    MelodyFromHumResponse,
    RefineSessionRequest,
    SessionChatRequest,
    SessionChatResponse,
    SessionCreateRequest,
    SessionMoodRequest,
    SessionMoodResponse,
    SessionStateResponse,
    SuggestLyricsRequest,
    SuggestLyricsResponse,
)
from backend.database import Database
from backend.experiment_logger import ExperimentLogger
from backend.logging_config import (
    bind_request_context,
    get_logger,
    get_request_context,
    initialize_request_context,
)
from backend.chat_responder import build_chat_reply
from backend.explain import build_explanation_report
from backend.history import add_history, append_chat_message
from backend.lyric_responder import generate_lyric_suggestions
from backend.mock_pipeline import (
    build_emotion_vector,
    build_progressions,
)
from backend.mood_responder import infer_mood
from backend.refinement_executor import execute_refinement_plan
from backend.refinement_parser import parse_refinement_instruction
from backend.session import SessionManager, SessionNotFoundError
from ml.gpt.pipeline import GPTPipeline, ResponseCache
from ml.harmony import generate_chords as generate_dqn_chords
from ml.harmony.quantize import notes_to_harmony_grid
from ml.melody_sketchpad.continuation.constraints import chord_symbol_to_pitch_classes
from ml.melody_sketchpad.continuation.pipeline import continue_melody as run_continuation_pipeline
from ml.melody_sketchpad.pipeline import NoMelodyDetectedError, run_melody_pipeline
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import (
    EMOTION_PRESET_LABELS,
    ChordProgression,
    EmotionVector,
    ExplanationPart,
    MelodySuggestion,
    NoteEvent,
    SessionState,
    SessionSummary,
    lookup_emotion_preset,
)

APP_VERSION = "0.3.0"

app = FastAPI(title="HumMuse API", version=APP_VERSION)
database = Database()
session_manager = SessionManager(database)
experiment_logger = ExperimentLogger()
request_logger = get_logger("backend.api")

_gpt_pipeline: GPTPipeline | None = None
_gpt_pipeline_last_failure_at: float | None = None
# How long to wait after a failed init before retrying. `from_env` only reads
# `.env`/process env (no network), so the usual failure is a missing or
# late-provisioned credential — a short cooldown lets the server recover once
# the env is fixed, without restarting, while avoiding a retry storm on every
# request. Override via HUMMUSE_GPT_INIT_RETRY_SECONDS (0 disables the cooldown).
_GPT_INIT_RETRY_SECONDS = 60.0


def _gpt_init_retry_seconds() -> float:
    raw = os.getenv("HUMMUSE_GPT_INIT_RETRY_SECONDS")
    if raw is None:
        return _GPT_INIT_RETRY_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _GPT_INIT_RETRY_SECONDS


def _get_gpt_pipeline() -> GPTPipeline | None:
    """Lazily construct a shared GPTPipeline; return None if config is missing.

    A failed init is remembered with a timestamp rather than latched
    permanently: subsequent calls reuse the "unavailable" verdict only until
    the retry cooldown elapses, then attempt construction again so a corrected
    credential recovers without a process restart.
    """
    global _gpt_pipeline, _gpt_pipeline_last_failure_at
    if _gpt_pipeline is not None:
        return _gpt_pipeline
    if _gpt_pipeline_last_failure_at is not None:
        cooldown = _gpt_init_retry_seconds()
        if cooldown > 0.0 and (time.monotonic() - _gpt_pipeline_last_failure_at) < cooldown:
            return None
    try:
        _gpt_pipeline = GPTPipeline.from_env(cache=ResponseCache(database))
    except Exception as exc:
        _gpt_pipeline_last_failure_at = time.monotonic()
        # `.error` (not `.info`): GPT being unavailable silently degrades every
        # GPT-backed endpoint to the mock, which is worth surfacing loudly. Use
        # error rather than warning because the structured-logger fallback
        # (_FallbackLogger) only implements info/error.
        request_logger.error(
            "gpt_pipeline_init_failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
            retry_after_seconds=_gpt_init_retry_seconds(),
        )
        return None
    _gpt_pipeline_last_failure_at = None
    return _gpt_pipeline


def _infer_pipeline_stage(endpoint: str) -> str:
    if endpoint.startswith("/session/"):
        if endpoint.endswith("/chat"):
            return "explanation"
        return "session"
    if endpoint.startswith("/melody/"):
        return "melody"
    if endpoint.startswith("/chords/"):
        return "harmony"
    if endpoint.startswith("/suggest/"):
        return "lyrics"
    if endpoint.startswith("/artifact/"):
        return "storage"
    return "api"


def _infer_use_case(endpoint: str) -> str | None:
    if endpoint == "/suggest/lyrics":
        return "lyric"
    if endpoint.endswith("/refine"):
        return "refine"
    if endpoint.endswith("/chat"):
        return "explain"
    return None


def _extract_path_session_id(endpoint: str) -> str | None:
    parts = [part for part in endpoint.split("/") if part]
    if len(parts) >= 2 and parts[0] == "session":
        return parts[1]
    return None


def _get_session_or_404(session_id: str) -> SessionState:
    try:
        return session_manager.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _save_session(state: SessionState) -> None:
    session_manager.update_session(state.session_id, state)


def _update_explanation_report(state: SessionState, *, source_action: str, summary: str, use_case: str | None = None) -> None:
    state.explanation_report = build_explanation_report(
        state,
        source_action=source_action,
        summary=summary,
        use_case=use_case,
    )


def _append_continuation_notes(state: SessionState, suggestion: MelodySuggestion) -> list[NoteEvent]:
    current_end = max((note.onset + note.duration for note in state.melody_notes), default=0.0)
    first_onset = min(note.onset for note in suggestion.notes)
    appended = [
        NoteEvent(
            pitch=note.pitch,
            onset=current_end + max(0.0, note.onset - first_onset),
            duration=note.duration,
            velocity=note.velocity,
            confidence=note.confidence,
        )
        for note in suggestion.notes
    ]
    state.melody_notes = [*state.melody_notes, *appended]
    state.melody_profile = build_melody_profile(state.melody_notes)
    state.melody_midi = _note_events_to_midi_bytes(state.melody_notes, tempo_bpm=int(state.detected_tempo or 120))
    return appended


def _note_events_to_midi_bytes(notes: list[NoteEvent], *, tempo_bpm: int) -> bytes:
    midi = MIDIFile(1, deinterleave=False)
    midi.addTrackName(0, 0, "HumMuse melody")
    midi.addTempo(0, 0, int(tempo_bpm))
    for note in sorted(notes, key=lambda item: (item.onset, item.duration, item.pitch)):
        midi.addNote(
            track=0,
            channel=0,
            pitch=int(note.pitch),
            time=float(note.onset),
            duration=float(note.duration),
            volume=int(note.velocity),
        )
    buffer = BytesIO()
    midi.writeFile(buffer)
    return buffer.getvalue()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    endpoint = request.url.path
    initialize_request_context(
        request,
        endpoint=endpoint,
        pipeline_stage=_infer_pipeline_stage(endpoint),
        session_id=_extract_path_session_id(endpoint),
        use_case=_infer_use_case(endpoint),
    )
    started_at = time.perf_counter()
    response = None
    error_detail = None

    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error_detail = str(exc)
        raise
    finally:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        context = get_request_context(request)
        success = response is not None and response.status_code < 400 and error_detail is None
        request_logger.info(
            "request_completed",
            **context,
            latency_ms=latency_ms,
            success=success,
            error=error_detail or (None if response is None or success else f"http_{response.status_code}"),
        )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": APP_VERSION}


@app.post("/session/create", response_model=SessionStateResponse)
def create_session(request: Request, payload: SessionCreateRequest | None = None) -> SessionStateResponse:
    request_payload = payload or SessionCreateRequest()
    state = session_manager.create_session()
    if request_payload.user_params:
        state.user_params.update(request_payload.user_params)
    if request_payload.lyrics_text:
        state.lyrics_text = request_payload.lyrics_text
    add_history(state, "session_created", {"user_params": bool(request_payload.user_params)})
    _save_session(state)
    bind_request_context(request, session_id=str(state.session_id), pipeline_stage="session")
    return SessionStateResponse(state=state)


@app.get("/session/{session_id}/state", response_model=SessionStateResponse)
def get_session_state(request: Request, session_id: str) -> SessionStateResponse:
    bind_request_context(request, session_id=session_id, pipeline_stage="session")
    return SessionStateResponse(state=_get_session_or_404(session_id))


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(request: Request) -> list[SessionSummary]:
    bind_request_context(request, pipeline_stage="session")
    return session_manager.list_sessions()


@app.post("/melody/from-hum", response_model=MelodyFromHumResponse)
async def melody_from_hum(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    prompt: str = Form("Melody sketch from humming"),
    mood: str | None = Form(default=None),
    tempo_bpm: int | None = Form(default=None),
) -> MelodyFromHumResponse:
    bind_request_context(request, session_id=session_id, pipeline_stage="melody")
    run_id = experiment_logger.start_run(
        "melody_from_hum",
        metadata={
            "filename": audio.filename,
            "prompt": prompt,
            "mood": mood,
            "tempo_bpm": tempo_bpm,
            "session_id": session_id,
        },
    )
    try:
        audio_bytes = await audio.read()
        melody_result = run_melody_pipeline(audio_bytes, tempo_bpm=tempo_bpm, mood=mood)
        melody = melody_result.melody
        note_events = melody_result.note_events
        melody_profile = melody_result.melody_profile
        detected_tempo = melody_result.detected_tempo
        base_url = str(request.base_url).rstrip("/")

        midi_ref = experiment_logger.log_artifact(
            run_id,
            kind="midi",
            filename="melody.mid",
            payload=melody_result.midi_bytes,
            base_url=base_url,
        )
        timing_ref = experiment_logger.log_artifact(
            run_id,
            kind="timings",
            filename="melody_pipeline_timings.json",
            payload={
                "step_latency_ms": melody_result.step_latency_ms,
                "metadata": melody_result.metadata,
            },
            base_url=base_url,
        )
        experiment_logger.finish_run(run_id)

        state = _get_session_or_404(session_id) if session_id else session_manager.create_session()
        bind_request_context(request, session_id=str(state.session_id), pipeline_stage="melody")
        state.melody_midi = melody_result.midi_bytes
        state.melody_notes = note_events
        state.melody_profile = melody_profile
        state.detected_key = melody_result.detected_key
        state.detected_tempo = detected_tempo
        # Chord generation is intentionally not coupled to melody upload —
        # callers invoke /chords/from-melody or /chords/from-lyrics explicitly
        # so they control when the DQN runs (and which emotion drives it).
        add_history(
            state,
            "melody_uploaded",
            {"filename": audio.filename or "audio", "tempo_bpm": tempo_bpm, "prompt": prompt},
        )
        melody_source = melody_result.metadata.get("melody_source", "unknown")
        if melody_source == "pesto":
            extraction_summary = (
                f"PESTO extracted {melody_result.metadata.get('raw_note_count', 0)} note events "
                f"(pitched ratio {melody_result.metadata.get('pitched_ratio', 0.0):.2f}); "
                "quantize+smooth produced the session melody profile."
            )
        elif melody_source == "pesto_rhythm_fallback":
            extraction_summary = (
                "Pitch confidence was low, so rhythm-only onsets were used to seed the melody profile."
            )
        else:
            extraction_summary = (
                f"Audio decoding fell back to a silent proxy (melody_source={melody_source}); "
                "a deterministic placeholder melody was returned so downstream stages stayed reproducible."
            )
        _update_explanation_report(
            state,
            source_action="melody_from_hum",
            summary=extraction_summary,
        )
        _save_session(state)

        return MelodyFromHumResponse(
            session_id=state.session_id,
            melody_notes=note_events,
            melody_profile=melody_profile,
            detected_key=state.detected_key,
            detected_tempo=detected_tempo,
            artifacts=[midi_ref, timing_ref],
            run_id=run_id,
            explanation=melody_result.explanation,
        )
    except NoMelodyDetectedError as exc:
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        # e.g. the 404 from _get_session_or_404, raised after the run has
        # already been finished successfully. Propagate the real status code
        # unchanged instead of letting the broad handler relabel it as a
        # failed 500 run (which would also double-finish the run).
        raise
    except Exception as exc:  # pragma: no cover
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


_NO_MELODY_FALLBACK_MAJOR = ["C", "G", "Am", "F"]
_NO_MELODY_FALLBACK_MINOR = ["Am", "F", "C", "G"]


@app.post("/chords/from-lyrics", response_model=LyricsToChordsResponse)
def chords_endpoint(request: Request, payload: LyricsToChordsRequest) -> LyricsToChordsResponse:
    """Generate chords from lyrics.

    Pipeline: lyrics → GPT mood inference (emotion vector + free-form,
    language-matched mood label) → DQN harmonizer over the session
    melody. Falls back to the keyword-bucket mood classifier when GPT is
    unavailable; falls back to a fixed two-progression library when the
    session has no melody (DQN cannot run without one).
    """
    bind_request_context(
        request,
        session_id=str(payload.session_id) if payload.session_id is not None else None,
        pipeline_stage="harmony",
    )

    mood_result = infer_mood(payload.text, pipeline=_get_gpt_pipeline())
    emotion_vector = mood_result.emotion_vector
    mood_label = mood_result.mood_label

    explanation: list[ExplanationPart] = [
        ExplanationPart(
            title="Mood inference",
            detail=(
                f"{'GPT' if mood_result.source == 'gpt' else 'Keyword fallback'} "
                f"read the lyrics as \"{mood_label}\" "
                f"(valence={emotion_vector.valence:+.2f}, arousal={emotion_vector.arousal:+.2f})."
            ),
        ),
    ]
    if mood_result.rationale:
        explanation.append(ExplanationPart(title="Why", detail=mood_result.rationale))

    response_mood = mood_label
    detected_mood: str | None = None
    detected_emotion_vector: EmotionVector | None = None
    mood_overridden = False
    mood_suggestion_pending = False

    if payload.session_id is not None:
        state = _get_session_or_404(str(payload.session_id))
        # Always record what inference saw, regardless of whether it's applied.
        state.user_params["last_mood_source"] = mood_result.source
        state.user_params["last_mood_label"] = mood_label
        state.user_params["last_mood_cache_hit"] = mood_result.cache_hit
        state.user_params["last_mood_error"] = mood_result.error

        authored = state.emotion_source == "authored" and state.emotion_vector is not None
        if authored:
            # Suggest, don't override: keep the author's Song Brief mood and
            # generate from it; surface the lyric-inferred mood as a suggestion.
            active_vector = state.emotion_vector
            response_mood = state.mood_label or mood_label
            detected_mood = mood_label
            detected_emotion_vector = emotion_vector
            mood_suggestion_pending = True
            state.user_params["suggested_mood_label"] = mood_label
            state.user_params["suggested_mood_valence"] = emotion_vector.valence
            state.user_params["suggested_mood_arousal"] = emotion_vector.arousal
            chord_progressions = _generate_chord_progressions_for_state(state, active_vector)
            explanation.append(
                ExplanationPart(
                    title="Kept your Song Brief mood",
                    detail=(
                        f"Detected mood '{mood_label}' is offered as a suggestion; your "
                        f"chosen Song Brief mood '{response_mood}' drives the harmony."
                    ),
                )
            )
            summary = (
                f"Lyrics read as '{mood_label}', but your Song Brief mood "
                f"'{response_mood}' was kept; DQN harmonizer produced "
                f"{len(chord_progressions)} progression candidate(s)."
            )
        else:
            # No authored mood — apply the inferred mood as the session emotion.
            mood_overridden = True
            active_vector = emotion_vector
            state.emotion_vector = emotion_vector
            state.mood_label = mood_label
            state.emotion_source = "detected"
            state.user_params["mood"] = mood_label
            chord_progressions = _generate_chord_progressions_for_state(state, active_vector)
            summary = (
                f"Lyric analysis inferred mood '{mood_label}' "
                f"(v={emotion_vector.valence:+.2f}, a={emotion_vector.arousal:+.2f}); "
                f"DQN harmonizer produced {len(chord_progressions)} progression candidate(s)."
            )

        state.lyrics_text = payload.text
        state.chord_progressions = chord_progressions
        add_history(
            state,
            "lyrics_analysed",
            {
                "mood_label": mood_label,
                "mood_source": mood_result.source,
                "valence": emotion_vector.valence,
                "arousal": emotion_vector.arousal,
                "harmonizer": "dqn" if state.melody_notes else "fallback",
                "mode": "suggested" if mood_suggestion_pending else "applied",
                "progression_count": len(chord_progressions),
            },
        )
        _update_explanation_report(
            state,
            source_action="chords_from_lyrics",
            summary=summary,
        )
        _save_session(state)
        top_progressions = [list(progression.chords) for progression in chord_progressions]
    else:
        chord_progressions = _build_no_melody_fallback_progressions(emotion_vector, mood_label)
        top_progressions = [list(progression.chords) for progression in chord_progressions]
        mood_overridden = True

    return LyricsToChordsResponse(
        mood=response_mood,
        top_progressions=top_progressions,
        chord_progressions=chord_progressions,
        explanation=explanation,
        session_id=payload.session_id,
        detected_mood=detected_mood,
        detected_emotion_vector=detected_emotion_vector,
        mood_overridden=mood_overridden,
        mood_suggestion_pending=mood_suggestion_pending,
    )


def _generate_chord_progressions_for_state(state: SessionState, emotion_vector) -> list:
    if not state.melody_notes:
        # No melody to drive the DQN — return the two-progression fallback
        # library shaped by the emotion vector's sign.
        return _build_no_melody_fallback_progressions(emotion_vector, mood_label_hint=None)
    # Snap a throwaway copy of the melody onto a beat grid for the chord model
    # only. The melody artifact the user keeps stays unquantized; this restores
    # the clean bar_position / duration_type channel the DQN needs so hummed
    # rubato does not collapse chord output. See ml/harmony/quantize.py.
    progressions = generate_dqn_chords(
        notes_to_harmony_grid(state.melody_notes),
        key=state.detected_key or "C major",
        emotion_vector=emotion_vector,
        top_k=3,
        tempo_bpm=state.detected_tempo or 120.0,
    )
    if not progressions:
        # The melody had notes but produced no usable chord events (e.g. all
        # rests/filtered by notes_to_events). Treat it like the no-melody case
        # so the response still satisfies the fixed 3-progression contract on
        # LyricsToChordsResponse.top_progressions instead of raising a 500.
        return _build_no_melody_fallback_progressions(emotion_vector, mood_label_hint=None)
    return progressions


def _build_no_melody_fallback_progressions(emotion_vector, mood_label_hint: str | None) -> list:
    """Two fixed progressions chosen by valence sign, returned with a
    transparent "fallback" explanation so callers know the DQN didn't run.
    """
    symbols = (
        _NO_MELODY_FALLBACK_MAJOR
        if emotion_vector.valence >= 0
        else _NO_MELODY_FALLBACK_MINOR
    )
    return build_progressions(
        [symbols, list(reversed(symbols)), symbols],
        mood_label_hint or "neutral",
    )


@app.post("/chords/from-melody", response_model=ChordsFromMelodyResponse)
def chords_from_melody_endpoint(request: Request, payload: ChordsFromMelodyRequest) -> ChordsFromMelodyResponse:
    """Run the DQN harmonizer over the session melody without any lyric input.

    Requires ``state.melody_notes``; returns 422 if the session has no melody.
    Emotion source priority: request body > ``state.emotion_vector`` > neutral
    default ``(valence=0, arousal=0.3)``. The DQN itself takes only the
    melody, key, and emotion vector — lyrics never enter this path.
    """
    bind_request_context(
        request,
        session_id=str(payload.session_id),
        pipeline_stage="harmony",
    )
    state = _get_session_or_404(str(payload.session_id))
    if not state.melody_notes:
        raise HTTPException(
            status_code=422,
            detail="Session has no melody_notes; upload a hum via /melody/from-hum first.",
        )

    emotion_vector = payload.emotion_vector or state.emotion_vector or EmotionVector(valence=0.0, arousal=0.3)
    run_id = experiment_logger.start_run(
        "chords_from_melody",
        metadata={
            "session_id": str(state.session_id),
            "top_k": payload.top_k,
            "emotion_vector": emotion_vector.model_dump(),
            "key": state.detected_key,
        },
    )
    try:
        # Harmonize a beat-grid-snapped copy (chord model only); the stored
        # melody stays unquantized. See ml/harmony/quantize.py.
        chord_progressions = generate_dqn_chords(
            notes_to_harmony_grid(state.melody_notes),
            key=state.detected_key or "C major",
            emotion_vector=emotion_vector,
            top_k=payload.top_k,
            tempo_bpm=state.detected_tempo or 120.0,
        )
        state.emotion_vector = emotion_vector
        state.chord_progressions = chord_progressions
        add_history(
            state,
            "chords_from_melody",
            {
                "top_k": payload.top_k,
                "emotion_vector": emotion_vector.model_dump(),
                "progression_count": len(chord_progressions),
                "run_id": run_id,
            },
        )
        _update_explanation_report(
            state,
            source_action="chords_from_melody",
            summary="DQN harmonizer generated chord progressions from the session melody and emotion vector (no lyrics).",
        )
        experiment_logger.finish_run(run_id)
        _save_session(state)
        return ChordsFromMelodyResponse(
            session_id=state.session_id,
            chord_progressions=chord_progressions,
            emotion_vector=emotion_vector,
            run_id=run_id,
        )
    except HTTPException:
        experiment_logger.finish_run(run_id)
        raise
    except Exception as exc:  # pragma: no cover - DQN/model failures
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/moods")
def list_moods() -> dict[str, list[str]]:
    """Return the canonical mood-preset labels (the Song Brief vocabulary).

    The UI fetches this to populate the Song Brief dropdown so the client and
    backend never drift on the mood taxonomy.
    """
    return {"labels": list(EMOTION_PRESET_LABELS)}


@app.post("/session/mood", response_model=SessionMoodResponse)
def session_mood_endpoint(request: Request, payload: SessionMoodRequest) -> SessionMoodResponse:
    """Persist an author-chosen Song Brief mood as the canonical session emotion.

    Sets ``state.emotion_vector`` from the preset, marks it ``emotion_source =
    "authored"``, and records the label. This is the single source of truth that
    cascades into the DQN harmonizer, the continuation temperature, and the GPT
    lyric prompt — all of which already read ``state.emotion_vector``.
    """
    bind_request_context(request, session_id=str(payload.session_id), pipeline_stage="session")
    state = _get_session_or_404(str(payload.session_id))
    vector = lookup_emotion_preset(payload.mood)
    if vector is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown mood '{payload.mood}'; must be one of: {', '.join(EMOTION_PRESET_LABELS)}.",
        )
    state.emotion_vector = vector
    state.mood_label = payload.mood
    state.emotion_source = "authored"
    # Mirror into user_params so the explanation report (reads user_params["mood"])
    # and the version slider keep working.
    state.user_params["mood"] = payload.mood
    state.user_params["last_mood_label"] = payload.mood
    state.user_params["last_mood_source"] = "authored"
    add_history(
        state,
        "mood_set",
        {
            "mood_label": payload.mood,
            "valence": vector.valence,
            "arousal": vector.arousal,
            "source": "authored",
        },
    )
    _update_explanation_report(
        state,
        source_action="session_mood",
        summary=(
            f"Song Brief set mood '{payload.mood}' "
            f"(v={vector.valence:+.2f}, a={vector.arousal:+.2f}); this drives DQN harmony "
            "bias, melody-continuation temperature, and the GPT lyric emotion vector."
        ),
    )
    _save_session(state)
    return SessionMoodResponse(
        session_id=state.session_id,
        mood_label=payload.mood,
        emotion_vector=vector,
        emotion_source="authored",
    )


@app.post("/chords/manual", response_model=ChordsManualResponse)
def chords_manual_endpoint(request: Request, payload: ChordsManualRequest) -> ChordsManualResponse:
    """Replace the session's chord progression with a user-supplied list.

    The progression is stored with `score=1.0` and `explanation="User-supplied
    progression."` so the Phase 5 chord-consonance constraint picks it up
    exactly like a DQN-accepted progression. We bypass the DQN/lyric pathway
    entirely — there is no mood inference and no model confidence to record.
    """
    bind_request_context(request, session_id=str(payload.session_id), pipeline_stage="harmony")
    cleaned: list[str] = []
    rejected: list[str] = []
    for symbol in payload.chords:
        cleaned_symbol = symbol.strip()
        if not cleaned_symbol:
            continue
        # chord_symbol_to_pitch_classes returns an empty set for anything it
        # cannot parse (unknown root, malformed suffix). That is exactly the
        # signal the constraint check would use downstream, so use it as the
        # validator here too — keep contracts in lock-step.
        if not chord_symbol_to_pitch_classes(cleaned_symbol):
            rejected.append(cleaned_symbol)
            continue
        cleaned.append(cleaned_symbol)
    if rejected:
        raise HTTPException(
            status_code=400,
            detail=f"Unparseable chord symbol(s): {', '.join(rejected)}",
        )
    if not cleaned:
        raise HTTPException(status_code=400, detail="At least one chord symbol is required")

    state = _get_session_or_404(str(payload.session_id))
    progression = ChordProgression(
        chords=cleaned,
        score=1.0,
        explanation="User-supplied progression.",
    )
    state.chord_progressions = [progression]
    add_history(
        state,
        "chords_manual",
        {"chord_count": len(cleaned), "chords": cleaned},
    )
    _update_explanation_report(
        state,
        source_action="chords_manual",
        summary="User supplied a chord progression directly; DQN/lyric pathway bypassed.",
    )
    _save_session(state)
    return ChordsManualResponse(session_id=payload.session_id, chord_progression=progression)


@app.post("/melody/continue", response_model=MelodyContinueResponse)
def continue_melody(request: Request, payload: MelodyContinueRequest) -> MelodyContinueResponse:
    bind_request_context(request, session_id=str(payload.session_id), pipeline_stage="melody")
    state = _get_session_or_404(str(payload.session_id))
    # Thread per-request section labels onto the loaded session before the
    # pipeline runs. Persisted on the session so subsequent /melody/accept and
    # explanation lookups can see what shaping was applied. None values are
    # honoured — they degrade the constraint stage to primer-relative checks.
    state.primer_section = payload.primer_section
    state.target_section = payload.target_section
    run_id = experiment_logger.start_run(
        "melody_continue",
        metadata={
            "session_id": str(state.session_id),
            "requested_suggestions": payload.num_suggestions,
            "engine": "music_transformer",
            "primer_section": payload.primer_section,
            "target_section": payload.target_section,
        },
    )
    try:
        state.melody_suggestions = run_continuation_pipeline(state, top_n=payload.num_suggestions)
        engines = sorted({suggestion.engine for suggestion in state.melody_suggestions})
        state.user_params["continuation_run_id"] = run_id
        state.user_params["continuation_engines"] = engines
        experiment_logger.log_artifact(
            run_id,
            kind="continuation_trace",
            filename="continuation_trace.json",
            payload={
                "candidate_count": state.user_params.get("continuation_candidate_count"),
                "survivor_count": state.user_params.get("continuation_survivor_count"),
                "constraint_trace": state.user_params.get("continuation_constraint_trace", []),
                "rejection_metadata": state.user_params.get("continuation_rejection_metadata", []),
                "engines": engines,
            },
            base_url=str(request.base_url).rstrip("/"),
        )
        experiment_logger.finish_run(run_id)
        add_history(
            state,
            "melody_continued",
            {"count": len(state.melody_suggestions), "engine": engines[0] if len(engines) == 1 else engines},
        )
        _update_explanation_report(
            state,
            source_action="melody_continue",
            summary="Music Transformer continuation generated candidate phrases with constraint traces and profile scores.",
        )
        _save_session(state)
        return MelodyContinueResponse(
            session_id=state.session_id,
            melody_suggestions=state.melody_suggestions,
        )
    except Exception as exc:  # pragma: no cover - model/runtime failures surface to API clients
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/melody/accept", response_model=SessionStateResponse)
def accept_melody_continuation(request: Request, payload: MelodyAcceptRequest) -> SessionStateResponse:
    bind_request_context(request, session_id=str(payload.session_id), pipeline_stage="melody")
    state = _get_session_or_404(str(payload.session_id))
    if payload.suggestion_index >= len(state.melody_suggestions):
        raise HTTPException(status_code=400, detail="suggestion_index is out of range")

    suggestion = state.melody_suggestions[payload.suggestion_index]
    if not suggestion.notes:
        raise HTTPException(status_code=400, detail="Selected continuation has no decoded notes")

    run_id = experiment_logger.start_run(
        "melody_continue_accept",
        metadata={
            "session_id": str(state.session_id),
            "suggestion_index": payload.suggestion_index,
            "engine": suggestion.engine,
            "coherence_score": suggestion.coherence_score,
        },
    )
    try:
        appended_notes = _append_continuation_notes(state, suggestion)
        state.user_params["accepted_continuation_engine"] = suggestion.engine
        state.user_params["accepted_continuation_index"] = payload.suggestion_index
        state.user_params["accepted_continuation_run_id"] = run_id
        state.melody_suggestions = []
        add_history(
            state,
            "melody_continuation_accepted",
            {
                "suggestion_index": payload.suggestion_index,
                "engine": suggestion.engine,
                "note_count": len(appended_notes),
                "run_id": run_id,
            },
        )
        _update_explanation_report(
            state,
            source_action="melody_continue_accept",
            summary="Selected Music Transformer continuation was appended to the session melody and the profile was recomputed.",
        )
        experiment_logger.finish_run(run_id)
        _save_session(state)
        return SessionStateResponse(state=state)
    except Exception as exc:  # pragma: no cover
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/suggest/lyrics", response_model=SuggestLyricsResponse)
def suggest_lyrics(request: Request, payload: SuggestLyricsRequest) -> SuggestLyricsResponse:
    bind_request_context(request, session_id=str(payload.session_id), pipeline_stage="lyrics", use_case="lyric")
    state = _get_session_or_404(str(payload.session_id))
    # Persist a client-supplied lyrics buffer first so the prompt sees
    # the latest text. Treat empty/whitespace-only as "no update"; only
    # an explicit non-empty value overwrites existing state.
    if payload.lyrics_text is not None and payload.lyrics_text.strip():
        state.lyrics_text = payload.lyrics_text
    result = generate_lyric_suggestions(
        state,
        mode=payload.mode,
        num_suggestions=payload.num_suggestions,
        pipeline=_get_gpt_pipeline(),
    )
    state.lyric_suggestions = result.suggestions
    state.user_params["last_lyric_source"] = result.source
    state.user_params["last_lyric_mode"] = result.mode
    state.user_params["last_lyric_model_version"] = result.model_version
    state.user_params["last_lyric_cache_hit"] = result.cache_hit
    state.user_params["last_lyric_latency_ms"] = result.latency_ms
    state.user_params["last_lyric_error"] = result.error
    add_history(
        state,
        "lyrics_suggested",
        {
            "count": payload.num_suggestions,
            "mode": payload.mode,
            "source": result.source,
            "error": result.error,
        },
    )
    summary = (
        "GPT lyric suggestion grounded its options in the active session context."
        if result.source == "gpt"
        else "Mock lyric suggestion generation used the active session context and lyric mode selector."
    )
    _update_explanation_report(
        state,
        source_action="suggest_lyrics",
        summary=summary,
        use_case="lyric",
    )
    _save_session(state)
    return SuggestLyricsResponse(
        session_id=state.session_id,
        lyric_suggestions=state.lyric_suggestions,
        source=result.source,
        mode=result.mode,
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        error=result.error,
    )


@app.patch("/session/{session_id}/refine", response_model=SessionStateResponse)
def refine_session(request: Request, session_id: str, payload: RefineSessionRequest) -> SessionStateResponse:
    bind_request_context(request, session_id=session_id, pipeline_stage="session", use_case="refine")
    state = _get_session_or_404(session_id)
    pipeline = _get_gpt_pipeline()
    parsed = parse_refinement_instruction(
        state,
        payload.instruction,
        target_hint=payload.target,
        pipeline=pipeline,
    )
    refinement_plan = parsed.plan
    execution = execute_refinement_plan(state, refinement_plan, pipeline=pipeline)

    state.user_params["last_refinement"] = payload.instruction
    state.user_params["last_refinement_target"] = payload.target or "session"
    state.user_params["last_refinement_plan"] = refinement_plan.model_dump()
    state.user_params["last_refinement_interpretation"] = refinement_plan.interpretation
    state.user_params["last_refinement_source"] = parsed.source
    state.user_params["last_refinement_model_version"] = parsed.model_version
    state.user_params["last_refinement_cache_hit"] = parsed.cache_hit
    state.user_params["last_refinement_latency_ms"] = parsed.latency_ms
    state.user_params["last_refinement_error"] = parsed.error
    state.user_params["last_refinement_execution"] = execution.to_dict()
    add_history(
        state,
        "session_refined",
        {
            "instruction": payload.instruction,
            "target": payload.target,
            "plan": refinement_plan.model_dump(),
            "source": parsed.source,
            "error": parsed.error,
            "execution": execution.to_dict(),
        },
    )
    _update_explanation_report(
        state,
        source_action="session_refine",
        summary=refinement_plan.interpretation,
        use_case="refine",
    )
    _save_session(state)
    return SessionStateResponse(state=state)


@app.post("/session/{session_id}/chat", response_model=SessionChatResponse)
def session_chat(request: Request, session_id: str, payload: SessionChatRequest) -> SessionChatResponse:
    bind_request_context(request, session_id=session_id, pipeline_stage="explanation", use_case="explain")
    state = _get_session_or_404(session_id)
    append_chat_message(state, "user", payload.message)
    chat_reply = build_chat_reply(
        state,
        payload.message,
        pipeline=_get_gpt_pipeline(),
        language=payload.language,
    )
    reply = chat_reply.message
    state.chat_history.append(reply)
    state.user_params["last_chat_source"] = chat_reply.source
    state.user_params["last_chat_model_version"] = chat_reply.model_version
    state.user_params["last_chat_cache_hit"] = chat_reply.cache_hit
    state.user_params["last_chat_latency_ms"] = chat_reply.latency_ms
    state.user_params["last_chat_limits"] = chat_reply.limits
    state.user_params["last_chat_error"] = chat_reply.error
    add_history(
        state,
        "session_chat",
        {
            "message": payload.message,
            "source": chat_reply.source,
            "limits": chat_reply.limits,
            "error": chat_reply.error,
        },
    )
    summary = (
        "GPT explanation dialog grounded the answer in the latest Tier 1 explanation report."
        if chat_reply.source == "gpt"
        else "Mock explanation dialog grounded its response in the latest structured explanation report."
    )
    _update_explanation_report(
        state,
        source_action="session_chat",
        summary=summary,
        use_case="explain",
    )
    _save_session(state)
    return SessionChatResponse(
        session_id=state.session_id,
        reply=reply,
        chat_history=state.chat_history,
        explanation_report=state.explanation_report,
        reply_source=chat_reply.source,
        reply_model_version=chat_reply.model_version,
        reply_cache_hit=chat_reply.cache_hit,
        reply_latency_ms=chat_reply.latency_ms,
        reply_limits=chat_reply.limits,
        reply_error=chat_reply.error,
    )


@app.get("/artifact/{artifact_id}", response_model=ArtifactListingResponse)
def get_artifact_listing(request: Request, artifact_id: str) -> ArtifactListingResponse:
    bind_request_context(request, pipeline_stage="storage")
    artifact_dir = experiment_logger.artifact_store.root_dir / artifact_id
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise HTTPException(status_code=404, detail="Artifact not found")
    files = sorted(path.name for path in artifact_dir.iterdir() if path.is_file())
    return ArtifactListingResponse(artifact_id=artifact_id, files=files)


@app.get("/artifact/{artifact_id}/{filename}")
def get_artifact(request: Request, artifact_id: str, filename: str) -> FileResponse:
    bind_request_context(request, pipeline_stage="storage")
    path = experiment_logger.artifact_store.get_artifact_path(artifact_id, filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)
