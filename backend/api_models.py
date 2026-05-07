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
    session_id: UUID | None = None


class MelodyContinueRequest(BaseModel):
    session_id: UUID
    num_suggestions: int = Field(default=3, ge=1, le=5)


class MelodyContinueResponse(BaseModel):
    session_id: UUID
    melody_suggestions: list[MelodySuggestion] = Field(default_factory=list)


class SuggestLyricsRequest(BaseModel):
    session_id: UUID
    mode: str = Field(default="continue", min_length=1)
    num_suggestions: int = Field(default=3, ge=1, le=5)


class SuggestLyricsResponse(BaseModel):
    session_id: UUID
    lyric_suggestions: list[LyricSuggestion] = Field(default_factory=list)


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


class ArtifactListingResponse(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    files: list[str] = Field(default_factory=list)


class SessionStateResponse(BaseModel):
    state: SessionState
