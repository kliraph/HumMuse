"""API request/response models for EPIC 2 endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from shared.schemas import (
    ArtifactRef,
    ChatMessage,
    EmotionVector,
    ExplanationReport,
    ExplanationPart,
    LyricSuggestion,
    MelodyProfile,
    MelodySuggestion,
    NoteEvent,
    Progression,
    Section,
    SessionState,
)


class SessionCreateRequest(BaseModel):
    user_params: dict[str, Any] = Field(default_factory=dict)
    lyrics_text: str | None = None


class MelodyFromHumResponse(BaseModel):
    session_id: UUID
    melody_notes: list[NoteEvent] = Field(default_factory=list)
    melody_profile: MelodyProfile | None = None
    detected_key: str | None = None
    detected_tempo: float | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    run_id: str | None = None
    explanation: list[ExplanationPart] = Field(default_factory=list)


class LyricsToChordsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: UUID | None = None


class LyricsToChordsResponse(BaseModel):
    mood: str
    top_progressions: list[list[str]] = Field(..., min_length=3, max_length=3)
    chord_progressions: list[Progression] = Field(default_factory=list)
    explanation: list[ExplanationPart] = Field(default_factory=list)
    # Mood provenance for the "suggest, don't override" flow. When the session
    # already has an author-chosen Song Brief mood, the lyric-inferred mood is
    # surfaced as a suggestion instead of overwriting the authored vector.
    detected_mood: str | None = None
    detected_emotion_vector: EmotionVector | None = None
    mood_overridden: bool = False
    mood_suggestion_pending: bool = False


class ChordsFromMelodyRequest(BaseModel):
    """Request body for melody-driven DQN chord generation.

    No lyrics involved — the caller has already uploaded a melody to the
    session and wants chords driven purely by the melody + an optional
    emotion. When ``emotion_vector`` is omitted, the endpoint uses
    ``state.emotion_vector`` if present, otherwise a neutral default
    (valence=0, arousal=0.3).
    """

    session_id: UUID
    emotion_vector: EmotionVector | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class ChordsFromMelodyResponse(BaseModel):
    session_id: UUID
    chord_progressions: list[Progression] = Field(default_factory=list)
    emotion_vector: EmotionVector
    run_id: str | None = None


class SessionMoodRequest(BaseModel):
    """Persist an author-chosen Song Brief mood as the canonical session emotion.

    ``mood`` must be one of the curated preset labels (EMOTION_PRESET_LABELS);
    the endpoint rejects anything else with 422.
    """

    session_id: UUID
    mood: str = Field(..., min_length=1)


class SessionMoodResponse(BaseModel):
    session_id: UUID
    mood_label: str
    emotion_vector: EmotionVector
    emotion_source: str


class ChordsManualRequest(BaseModel):
    """User-supplied chord progression that bypasses the DQN/lyric pathway.

    `chords` is the already-tokenised list of chord symbols
    (e.g. ["C", "F", "G", "C"]). The UI handles splitting on commas/arrows/
    whitespace and submits the cleaned list.
    """

    session_id: UUID
    chords: list[str] = Field(..., min_length=1, max_length=64)


class ChordsManualResponse(BaseModel):
    session_id: UUID
    chord_progression: Progression


class MelodyContinueRequest(BaseModel):
    session_id: UUID
    num_suggestions: int = Field(default=3, ge=1, le=5)
    # Optional song-form labels. When both are set and the BiMMuDa transition
    # prior for (primer_section -> target_section) is statistically reliable,
    # the continuation pipeline switches to section-conditional constraints.
    # When either is None, the pipeline degrades to primer-relative checks
    # (continue stylistically near the primer).
    primer_section: Section | None = None
    target_section: Section | None = None


class MelodyContinueResponse(BaseModel):
    session_id: UUID
    melody_suggestions: list[MelodySuggestion] = Field(default_factory=list)


class MelodyAcceptRequest(BaseModel):
    session_id: UUID
    suggestion_index: int = Field(..., ge=0)


class SuggestLyricsRequest(BaseModel):
    session_id: UUID
    mode: str = Field(default="continue", min_length=1)
    num_suggestions: int = Field(default=3, ge=1, le=5)
    # Optional: if the client carries a typed lyrics buffer (e.g. the
    # Lyrics tab text area), persist it to session state before
    # generating so the GPT prompt sees the latest text. Backward-
    # compatible — older clients omit this and the endpoint reads
    # whatever already lives on the session.
    lyrics_text: str | None = None


class SuggestLyricsResponse(BaseModel):
    session_id: UUID
    lyric_suggestions: list[LyricSuggestion] = Field(default_factory=list)
    source: str = Field(default="fallback")
    mode: str | None = None
    model_version: str | None = None
    cache_hit: bool = False
    latency_ms: float | None = None
    error: str | None = None


class RefineSessionRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    target: str | None = None


class SessionChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    language: str | None = None


class SessionChatResponse(BaseModel):
    session_id: UUID
    reply: ChatMessage
    chat_history: list[ChatMessage] = Field(default_factory=list)
    explanation_report: ExplanationReport | None = None
    reply_source: str = Field(default="fallback")
    reply_model_version: str | None = None
    reply_cache_hit: bool = False
    reply_latency_ms: float | None = None
    reply_limits: str | None = None
    reply_error: str | None = None


class ArtifactListingResponse(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    files: list[str] = Field(default_factory=list)


class SessionStateResponse(BaseModel):
    state: SessionState
