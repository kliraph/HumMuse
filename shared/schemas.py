"""Pydantic schemas shared between backend and ML layers."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]


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


class MelodySuggestion(SchemaModel):
    midi_bytes: bytes = Field(...)
    notes: list[NoteEvent] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    coherence_score: float = Field(..., ge=0, le=1)


class LyricSuggestion(SchemaModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    syllable_count: int = Field(..., ge=0)


_PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PITCH_CLASS_MEMBERSHIP_LABELS = [*_PITCH_CLASS_LABELS, "triad_sentinel"]


def _top_margin(values: list[float]) -> float:
    if not values:
        return 0.0
    ranked = sorted((float(value) for value in values), reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    return ranked[0] - ranked[1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _advantages(values: list[float]) -> tuple[float, list[float]]:
    value = _mean(values)
    return value, [float(item) - value for item in values]


def _distribution_defaults(data: dict[str, Any]) -> dict[str, Any]:
    if "policy" not in data:
        rest = dict(data.get("rest", {}))
        octave = dict(data.get("octave", {}))
        inversion = dict(data.get("inversion", {}))
        pitch_class = dict(data.get("pitch_class", {}))
        membership = list(data.get("pitch_class_membership", []))
        labels = list(data.get("pitch_class_labels", _PITCH_CLASS_MEMBERSHIP_LABELS))
        data["policy"] = {
            "rest": rest,
            "octave": octave,
            "inversion": inversion,
            "pitch_class": pitch_class,
            "pitch_class_membership": membership,
            "pitch_class_labels": labels,
        }

    policy = dict(data.get("policy", {}))
    position = int(data.get("position", 0))
    rest = dict(policy.get("rest", {}))
    octave = dict(policy.get("octave", {}))
    inversion = dict(policy.get("inversion", {}))
    membership = list(policy.get("pitch_class_membership", []))

    if "q_values" not in data:
        data["q_values"] = {
            "rest": float(rest.get("rest", 0.0)),
            "octave": [float(octave.get(label, 0.0)) for label in ("2", "3")],
            "inversion": [float(inversion.get(str(index), 0.0)) for index in range(4)],
            "pitch_class": [float(value) for value in membership],
        }

    q_values = dict(data.get("q_values", {}))
    octave_values = [float(value) for value in q_values.get("octave", [])]
    inversion_values = [float(value) for value in q_values.get("inversion", [])]
    pitch_class_values = [float(value) for value in q_values.get("pitch_class", [])]
    rest_q = float(q_values.get("rest", 0.0))

    if "q_margin" not in data:
        data["q_margin"] = {
            "rest": _top_margin([rest_q, 1.0 - rest_q]),
            "octave": _top_margin(octave_values),
            "inversion": _top_margin(inversion_values),
            "pitch_class": _top_margin(pitch_class_values),
        }

    if "dueling" not in data:
        value_rest, advantage_rest = _advantages([rest_q, 1.0 - rest_q])
        value_octave, advantage_octave = _advantages(octave_values)
        value_inversion, advantage_inversion = _advantages(inversion_values)
        value_pc, advantage_pc = _advantages(pitch_class_values)
        data["dueling"] = {
            "value_rest": value_rest,
            "value_octave": value_octave,
            "value_inversion": value_inversion,
            "value_pc": value_pc,
            "advantage_rest": advantage_rest,
            "advantage_octave": advantage_octave,
            "advantage_inversion": advantage_inversion,
            "advantage_pc": advantage_pc,
        }

    data.setdefault(
        "context",
        {
            "prev_chord_pcs": [],
            "prev_chord_symbol": None,
            "current_note": {"pitch": 0, "duration": 0, "position": position},
        },
    )
    data.setdefault("reward_attribution", None)
    data.setdefault("emotion_bias", None)
    data.setdefault("model_kind", "dqn")
    data.setdefault("noise_state", "disabled")
    data.setdefault("noise_samples", None)
    return data


class ChordPolicy(SchemaModel):
    rest: dict[str, Probability] = Field(default_factory=dict)
    octave: dict[str, Probability] = Field(default_factory=dict)
    inversion: dict[str, Probability] = Field(default_factory=dict)
    pitch_class: dict[str, Probability] = Field(default_factory=dict)
    pitch_class_membership: list[Probability] = Field(..., min_length=13, max_length=13)
    pitch_class_labels: list[str] = Field(..., min_length=13, max_length=13)


class ChordQValues(SchemaModel):
    rest: float
    octave: list[float] = Field(default_factory=list)
    inversion: list[float] = Field(default_factory=list)
    pitch_class: list[float] = Field(default_factory=list)


class ChordQMargin(SchemaModel):
    rest: float
    octave: float
    inversion: float
    pitch_class: float


class ChordDueling(SchemaModel):
    value_rest: float
    value_octave: float
    value_inversion: float
    value_pc: float
    advantage_rest: list[float] = Field(default_factory=list)
    advantage_octave: list[float] = Field(default_factory=list)
    advantage_inversion: list[float] = Field(default_factory=list)
    advantage_pc: list[float] = Field(default_factory=list)


class ChordContextNote(SchemaModel):
    pitch: int = 0
    duration: int = 0
    position: int = 0


class ChordContext(SchemaModel):
    prev_chord_pcs: list[int] = Field(default_factory=list)
    prev_chord_symbol: str | None = None
    current_note: ChordContextNote


class ChordRewardAttribution(SchemaModel):
    harmony_rule: float
    progression_penalty: float
    chord_tone_inclusion: float
    mutual_info: float
    key_fit_proxy: float


class ChordEmotionBias(SchemaModel):
    applied: bool
    q_delta_per_pc: list[float] = Field(default_factory=list)
    rationale: str


class SelectedChord(SchemaModel):
    is_rest: bool
    octave: int | None = None
    inversion: int | None = None
    pcs: list[int] = Field(default_factory=list)
    pc_names: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ChordDistribution(SchemaModel):
    """Serializable DQN chord-head output for one chord position."""

    position: int = Field(..., ge=0)
    policy: ChordPolicy
    q_values: ChordQValues
    q_margin: ChordQMargin
    dueling: ChordDueling
    context: ChordContext
    reward_attribution: ChordRewardAttribution | None = None
    emotion_bias: ChordEmotionBias | None = None
    selected: SelectedChord
    model_kind: Literal["dqn"] = "dqn"
    noise_state: Literal["disabled", "enabled"] = "disabled"
    noise_samples: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_flat_distribution(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _distribution_defaults(dict(data))
        return data

    @property
    def rest(self) -> dict[str, Probability]:
        return self.policy.rest

    @property
    def octave(self) -> dict[str, Probability]:
        return self.policy.octave

    @property
    def inversion(self) -> dict[str, Probability]:
        return self.policy.inversion

    @property
    def pitch_class(self) -> dict[str, Probability]:
        return self.policy.pitch_class

    @property
    def pitch_class_membership(self) -> list[Probability]:
        return self.policy.pitch_class_membership

    @property
    def pitch_class_labels(self) -> list[str]:
        return self.policy.pitch_class_labels


class ChordSymbolDerivation(SchemaModel):
    """Deterministic chord-symbol reading derived from one native DQN output."""

    symbol: str = Field(..., min_length=1)
    root: str = Field(..., min_length=1)
    root_pc: int = Field(..., ge=0, le=11)
    quality: str = Field(..., min_length=1)
    bass: str = Field(..., min_length=1)
    bass_pc: int = Field(..., ge=0, le=11)
    pitch_classes: list[int] = Field(default_factory=list)
    inversion: int = Field(..., ge=0, le=3)
    roman_numeral: str = Field(..., min_length=1)
    harmonic_function: str = Field(..., min_length=1)
    confidence: Probability = 0.0
    candidate_symbols: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class ChordAnnotation(SchemaModel):
    """Tier 1 deterministic XAI metadata for one generated chord position."""

    position: int = Field(..., ge=0)
    symbol: str = Field(..., min_length=1)
    root: str = Field(..., min_length=1)
    quality: str = Field(..., min_length=1)
    bass: str = Field(..., min_length=1)
    roman_numeral: str = Field(..., min_length=1)
    function_label: str = Field(..., min_length=1)
    alignment_percentage: Probability = 0.0
    q_margin: float | None = None
    value_score: float | None = None
    strong_beat_notes: list[dict[str, Any]] = Field(default_factory=list)
    template_phrase: str = Field(..., min_length=1)
    derivation: ChordSymbolDerivation


class ChordExplanation(SchemaModel):
    """UI-facing explanation for one selected chord decision."""

    position: int = Field(..., ge=0)
    chord_symbol: str = Field(..., min_length=1)
    q_chosen: float
    q_runner_up: float
    runner_up_symbol: str
    margin: float
    value_score: float
    advantage_score: float
    prev_chord_link: str | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    emotion_bias_summary: str | None = None
    prose: str = Field(..., min_length=1)


class ChordProgression(SchemaModel):
    chords: list[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    mood_alignment: float | None = Field(default=None, ge=0, le=1)
    harmonic_function: str | None = None
    explanation: str = Field(..., min_length=1)
    native_distributions: list[ChordDistribution] = Field(default_factory=list)
    chord_annotations: list[ChordAnnotation] = Field(default_factory=list)
    chord_explanations: list[ChordExplanation] = Field(default_factory=list)


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

