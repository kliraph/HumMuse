"""API request/response models for EPIC 2 endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shared.schemas import ExplanationPart


class LyricsToChordsRequest(BaseModel):
    text: str = Field(..., min_length=1)


class LyricsToChordsResponse(BaseModel):
    mood: str
    top_progressions: list[list[str]] = Field(..., min_length=3, max_length=3)
    explanation: list[ExplanationPart] = Field(default_factory=list)


class WriterBlockHelpRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mood: str | None = None


class WriterBlockHelpResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list)
    explanation: list[ExplanationPart] = Field(default_factory=list)

