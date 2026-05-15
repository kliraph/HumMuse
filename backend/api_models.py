"""API request/response models for EPIC 2 endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from shared.schemas import (
    ArtifactRef,
    ChatMessage,
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
    chord_progressions: list[Progression] = Field(default_factory=list)
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
    session_id: UUID | None = None


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


class SuggestLyricsResponse(BaseModel):
    session_id: UUID
    lyric_suggestions: list[LyricSuggestion] = Field(default_factory=list)
    source: str = Field(default="fallback")
    mode: str | None = None
    model_version: str | None = None
    cache_hit: bool = False
    latency_ms: float | None = None
    error: str | None = None


class WriterBlockHelpRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mood: str | None = None


class WriterBlockHelpResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list)
    explanation: list[ExplanationPart] = Field(default_factory=list)


class RefineSessionRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    target: str | None = None


class SessionChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


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
