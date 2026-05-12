"""FastAPI app that orchestrates backend generation flows."""

from __future__ import annotations

import mimetypes
import time
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from midiutil import MIDIFile

from backend.api_models import (
    ArtifactListingResponse,
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
    SessionStateResponse,
    SuggestLyricsRequest,
    SuggestLyricsResponse,
    WriterBlockHelpRequest,
    WriterBlockHelpResponse,
)
from backend.database import Database
from backend.experiment_logger import ExperimentLogger
from backend.logging_config import (
    bind_request_context,
    get_logger,
    get_request_context,
    initialize_request_context,
)
from backend.mock_pipeline import (
    add_history,
    append_chat_message,
    build_chat_reply,
    build_emotion_vector,
    build_explanation_report,
    build_lyric_suggestions,
    build_progressions,
    build_refinement_plan,
)
from backend.session import SessionManager, SessionNotFoundError
from ml.harmony import generate_chords as generate_dqn_chords
from ml.melody_sketchpad.continuation.constraints import chord_symbol_to_pitch_classes
from ml.lyrics_to_chords.service import generate_from_lyrics
from ml.melody_sketchpad.continuation.pipeline import continue_melody as run_continuation_pipeline
from ml.melody_sketchpad.pipeline import run_melody_pipeline
from ml.melody_sketchpad.profile import build_melody_profile
from ml.writers_block.service import generate_help
from shared.schemas import ChordProgression, MelodySuggestion, NoteEvent, SessionState, SessionSummary

APP_VERSION = "0.3.0"

app = FastAPI(title="HumMuse API", version=APP_VERSION)
database = Database()
session_manager = SessionManager(database)
experiment_logger = ExperimentLogger()
request_logger = get_logger("backend.api")


def _infer_pipeline_stage(endpoint: str) -> str:
    if endpoint.startswith("/session/"):
        if endpoint.endswith("/chat"):
            return "explanation"
        return "session"
    if endpoint.startswith("/melody/"):
        return "melody"
    if endpoint.startswith("/chords/"):
        return "harmony"
    if endpoint.startswith("/suggest/") or endpoint.startswith("/writerblock/"):
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
        chord_progressions = build_progressions([[chord.symbol for chord in melody_result.chords]], mood or "uplift")
        base_url = str(request.base_url).rstrip("/")

        midi_ref = experiment_logger.log_artifact(
            run_id,
            kind="midi",
            filename="melody.mid",
            payload=melody_result.midi_bytes,
            base_url=base_url,
        )
        chord_ref = experiment_logger.log_artifact(
            run_id,
            kind="chords",
            filename="chords.json",
            payload={"chords": [c.model_dump() for c in melody_result.chords]},
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
        state.chord_progressions = chord_progressions
        add_history(
            state,
            "melody_uploaded",
            {"filename": audio.filename or "audio", "tempo_bpm": tempo_bpm, "prompt": prompt},
        )
        _update_explanation_report(
            state,
            source_action="melody_from_hum",
            summary="Mock melody extraction produced confidence-tagged notes and a session melody profile.",
        )
        _save_session(state)

        return MelodyFromHumResponse(
            session_id=state.session_id,
            melody_notes=note_events,
            melody_profile=melody_profile,
            detected_key=state.detected_key,
            detected_tempo=detected_tempo,
            chord_progressions=chord_progressions,
            artifacts=[midi_ref, chord_ref, timing_ref],
            run_id=run_id,
            explanation=melody_result.explanation,
        )
    except Exception as exc:  # pragma: no cover
        experiment_logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chords/from-lyrics", response_model=LyricsToChordsResponse)
def chords_endpoint(request: Request, payload: LyricsToChordsRequest) -> LyricsToChordsResponse:
    mood, top_progressions, explanation = generate_from_lyrics(payload.text)
    bind_request_context(
        request,
        session_id=str(payload.session_id) if payload.session_id is not None else None,
        pipeline_stage="harmony",
    )
    if payload.session_id is not None:
        state = _get_session_or_404(str(payload.session_id))
        emotion_vector = build_emotion_vector(mood)
        chord_progressions = _generate_chord_progressions_for_state(state, mood, top_progressions, emotion_vector)
        state.lyrics_text = payload.text
        state.emotion_vector = emotion_vector
        state.chord_progressions = chord_progressions
        add_history(
            state,
            "lyrics_analysed",
            {
                "mood": mood,
                "source": "dqn" if state.melody_notes else "mock",
                "progression_count": len(chord_progressions),
            },
        )
        _update_explanation_report(
            state,
            source_action="chords_from_lyrics",
            summary="Lyric analysis produced emotion-conditioned DQN chord progression candidates.",
        )
        _save_session(state)
        top_progressions = [progression.chords for progression in chord_progressions]
    else:
        chord_progressions = build_progressions(top_progressions, mood)
    return LyricsToChordsResponse(
        mood=mood,
        top_progressions=top_progressions,
        chord_progressions=chord_progressions,
        explanation=explanation,
        session_id=payload.session_id,
    )


def _generate_chord_progressions_for_state(
    state: SessionState,
    mood: str,
    fallback_progressions: list[list[str]],
    emotion_vector,
) -> list:
    if not state.melody_notes:
        return build_progressions(fallback_progressions, mood)
    return generate_dqn_chords(
        state.melody_notes,
        key=state.detected_key or "C major",
        emotion_vector=emotion_vector,
        top_k=3,
        tempo_bpm=state.detected_tempo or 120.0,
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
    state.lyric_suggestions = build_lyric_suggestions(state, payload.mode, payload.num_suggestions)
    add_history(state, "lyrics_suggested", {"count": payload.num_suggestions, "mode": payload.mode})
    _update_explanation_report(
        state,
        source_action="suggest_lyrics",
        summary="Mock lyric suggestion generation used the active session context and lyric mode selector.",
        use_case="lyric",
    )
    _save_session(state)
    return SuggestLyricsResponse(
        session_id=state.session_id,
        lyric_suggestions=state.lyric_suggestions,
    )


@app.patch("/session/{session_id}/refine", response_model=SessionStateResponse)
def refine_session(request: Request, session_id: str, payload: RefineSessionRequest) -> SessionStateResponse:
    bind_request_context(request, session_id=session_id, pipeline_stage="session", use_case="refine")
    state = _get_session_or_404(session_id)
    refinement_plan = build_refinement_plan(payload.instruction, payload.target)
    state.user_params["last_refinement"] = payload.instruction
    state.user_params["last_refinement_target"] = payload.target or "session"
    state.user_params["last_refinement_plan"] = refinement_plan.model_dump()
    state.user_params["last_refinement_interpretation"] = refinement_plan.interpretation
    add_history(
        state,
        "session_refined",
        {"instruction": payload.instruction, "target": payload.target, "plan": refinement_plan.model_dump()},
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
    reply = build_chat_reply(state, payload.message)
    state.chat_history.append(reply)
    add_history(state, "session_chat", {"message": payload.message})
    _update_explanation_report(
        state,
        source_action="session_chat",
        summary="Mock explanation dialog grounded its response in the latest structured explanation report.",
        use_case="explain",
    )
    _save_session(state)
    return SessionChatResponse(
        session_id=state.session_id,
        reply=reply,
        chat_history=state.chat_history,
        explanation_report=state.explanation_report,
    )


@app.post("/writerblock/help", response_model=WriterBlockHelpResponse)
def writerblock_endpoint(request: Request, payload: WriterBlockHelpRequest) -> WriterBlockHelpResponse:
    bind_request_context(request, pipeline_stage="lyrics")
    suggestions, explanation = generate_help(payload.text, payload.mood)
    return WriterBlockHelpResponse(suggestions=suggestions, explanation=explanation)


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
