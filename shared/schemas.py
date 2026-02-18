"""Pydantic schemas shared between backend and ML layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MelodyNote(BaseModel):
    pitch: str = Field(..., examples=["C4", "A#3"])
    start_beat: float = Field(..., ge=0)
    duration_beats: float = Field(..., gt=0)
    velocity: int = Field(default=100, ge=1, le=127)


class ChordSuggestion(BaseModel):
    symbol: str = Field(..., examples=["Am", "Fmaj7", "G7"])
    start_bar: int = Field(..., ge=0)
    beats: float = Field(..., gt=0)


class ExplanationPart(BaseModel):
    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)


class ArtifactRef(BaseModel):
    artifact_id: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    url: str | None = None


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mood: str | None = None
    tempo_bpm: int | None = Field(default=None, ge=40, le=240)
    melody: list[MelodyNote] = Field(default_factory=list)
    preferred_chords: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    run_id: str = Field(..., min_length=1)
    melody: list[MelodyNote] = Field(default_factory=list)
    chords: list[ChordSuggestion] = Field(default_factory=list)
    explanation: list[ExplanationPart] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)

