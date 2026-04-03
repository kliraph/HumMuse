"""Pydantic schemas shared between backend and ML layers."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")


class MelodyNote(SchemaModel):
    pitch: str = Field(..., examples=["C4", "A#3"])
    start_beat: float = Field(..., ge=0)
    duration_beats: float = Field(..., gt=0)
    velocity: int = Field(default=100, ge=1, le=127)


class NoteEvent(SchemaModel):
    pitch: int = Field(..., ge=0, le=127)
    onset: float = Field(..., ge=0)
    duration: float = Field(..., gt=0)
    velocity: int = Field(..., ge=1, le=127)
    confidence: float = Field(..., ge=0, le=1)


class MelodyProfile(SchemaModel):
    interval_histogram: list[float] = Field(default_factory=list)
    rhythmic_density: float = Field(..., ge=0)
    pitch_range: tuple[int, int] = Field(...)
    contour: Literal["rising", "falling", "arch", "valley"] = Field(...)


class EmotionVector(SchemaModel):
    valence: float = Field(..., ge=-1, le=1)
    arousal: float = Field(..., ge=-1, le=1)


class RecognisedChord(SchemaModel):
    symbol: str = Field(..., min_length=1, examples=["Am", "Fmaj7", "G7"])
    start_beat: float = Field(..., ge=0)
    duration_beats: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)
    harmonic_function: str | None = None
    explanation: str | None = None
    decoding_trace: list[dict[str, Any]] = Field(default_factory=list)


class MelodySuggestion(SchemaModel):
    midi_bytes: bytes = Field(...)
    notes: list[NoteEvent] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    coherence_score: float = Field(..., ge=0, le=1)


class LyricSuggestion(SchemaModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    syllable_count: int = Field(..., ge=0)


class ChordProgression(SchemaModel):
    chords: list[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1)
    harmonic_function: str | None = None
    explanation: str = Field(..., min_length=1)


class RefinementPlan(SchemaModel):
    target_pipeline: Literal["harmonizer", "melody_generator", "both", "lyric_generator", "session"]
    parameter_adjustments: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = Field(..., min_length=1)


class ChatMessage(SchemaModel):
    role: Literal["system", "user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1)
    timestamp: str | None = None


class ExplanationReport(SchemaModel):
    source_action: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    melody_confidence: list[dict[str, Any]] = Field(default_factory=list)
    decoding_traces: list[dict[str, Any]] = Field(default_factory=list)
    constraint_logs: list[dict[str, Any]] = Field(default_factory=list)
    chord_theory: list[dict[str, Any]] = Field(default_factory=list)
    emotion_mapping: dict[str, Any] = Field(default_factory=dict)
    cache_status: dict[str, Any] = Field(default_factory=dict)


class Action(SchemaModel):
    kind: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
    source: str | None = None


class ChordSuggestion(SchemaModel):
    symbol: str = Field(..., examples=["Am", "Fmaj7", "G7"])
    start_bar: int = Field(..., ge=0)
    beats: float = Field(..., gt=0)


class ExplanationPart(SchemaModel):
    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)


class ArtifactRef(SchemaModel):
    artifact_id: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    url: str | None = None


class GenerationRequest(SchemaModel):
    prompt: str = Field(..., min_length=1)
    mood: str | None = None
    tempo_bpm: int | None = Field(default=None, ge=40, le=240)
    melody: list[MelodyNote] = Field(default_factory=list)
    preferred_chords: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(SchemaModel):
    run_id: str = Field(..., min_length=1)
    melody: list[MelodyNote] = Field(default_factory=list)
    chords: list[ChordSuggestion] = Field(default_factory=list)
    explanation: list[ExplanationPart] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class SessionState(SchemaModel):
    session_id: UUID
    melody_midi: bytes | None = None
    melody_notes: list[NoteEvent] = Field(default_factory=list)
    melody_profile: MelodyProfile | None = None
    detected_key: str | None = None
    detected_tempo: float | None = Field(default=None, gt=0)
    recognised_chords: list[RecognisedChord] = Field(default_factory=list)
    chord_progressions: list[ChordProgression] = Field(default_factory=list)
    lyrics_text: str | None = None
    emotion_vector: EmotionVector | None = None
    melody_suggestions: list[MelodySuggestion] = Field(default_factory=list)
    lyric_suggestions: list[LyricSuggestion] = Field(default_factory=list)
    explanation_report: ExplanationReport | None = None
    chat_history: list[ChatMessage] = Field(default_factory=list)
    user_params: dict[str, Any] = Field(default_factory=dict)
    history: list[Action] = Field(default_factory=list)


class SessionSummary(SchemaModel):
    session_id: UUID
    created_at: str
    updated_at: str
    pipeline_version: str = Field(..., min_length=1)


Progression = ChordProgression
