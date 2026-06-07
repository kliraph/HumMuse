"""Pydantic schemas shared between backend and ML layers."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]

# Song-form section labels exposed to the continuation pipeline. The literal
# is kept tight (verse/pre_chorus/chorus/bridge) because BiMMuDa-derived
# transition priors are only reliable for these; instrumental and post_chorus
# are deliberately excluded from the UI.
Section = Literal["verse", "pre_chorus", "chorus", "bridge"]


class SchemaModel(BaseModel):
    # extra="forbid": unknown fields are a validation error, not silently
    # dropped. These models are the shared producer/consumer contract across
    # backend/ml/ui; rejecting stray keys surfaces drift (a renamed/removed
    # field, a typo'd key) at the boundary instead of letting data vanish on a
    # model_dump → model_validate round-trip.
    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
        extra="forbid",
    )


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


# --- Canonical mood/emotion taxonomy ---------------------------------------
#
# The single source of truth for the system's mood vocabulary. Every consumer
# — the UI Song Brief dropdown (via GET /moods), the backend refinement ops,
# the GPT-down keyword fallback, and the /session/mood endpoint — uses these
# 12 presets. Each maps to a curated point on the valence/arousal circumplex.
#
# Layout on the valence/arousal plane:
#     joyful       v=+0.8 a=+0.7   bright, energetic
#     triumphant   v=+0.9 a=+0.8   intense joy (joyful peak)
#     uplift       v=+0.6 a=+0.5   hopeful, rising
#     hopeful      v=+0.4 a=+0.3   gentle optimism
#     romantic     v=+0.5 a=+0.2   warm, intimate
#     calm         v=+0.3 a=-0.4   peaceful, low energy
#     neutral      v=+0.0 a=+0.0   no signal
#     reflective   v=-0.1 a=-0.3   contemplative, still
#     melancholic  v=-0.6 a=-0.2   sad, low energy
#     dark         v=-0.7 a=+0.1   heavy, slow burn
#     anxious      v=-0.4 a=+0.7   uneasy, high energy
#     tense        v=-0.3 a=+0.6   tight, charged
_EMOTION_PRESETS: dict[str, EmotionVector] = {
    "joyful":      EmotionVector(valence=0.8,  arousal=0.7),
    "triumphant":  EmotionVector(valence=0.9,  arousal=0.8),
    "uplift":      EmotionVector(valence=0.6,  arousal=0.5),
    "hopeful":     EmotionVector(valence=0.4,  arousal=0.3),
    "romantic":    EmotionVector(valence=0.5,  arousal=0.2),
    "calm":        EmotionVector(valence=0.3,  arousal=-0.4),
    "neutral":     EmotionVector(valence=0.0,  arousal=0.0),
    "reflective":  EmotionVector(valence=-0.1, arousal=-0.3),
    "melancholic": EmotionVector(valence=-0.6, arousal=-0.2),
    "dark":        EmotionVector(valence=-0.7, arousal=0.1),
    "anxious":     EmotionVector(valence=-0.4, arousal=0.7),
    "tense":       EmotionVector(valence=-0.3, arousal=0.6),
}

# Canonical display order for the UI dropdown (positive→neutral→negative).
EMOTION_PRESET_LABELS: tuple[str, ...] = tuple(_EMOTION_PRESETS)

_DEFAULT_EMOTION_VECTOR = EmotionVector(valence=0.1, arousal=0.4)

# Legacy/aliased labels that older UI builds or persisted sessions may carry,
# mapped onto a canonical preset key.
_MOOD_LABEL_ALIASES: dict[str, str] = {
    "melancholy": "melancholic",
}


def lookup_emotion_preset(mood: str) -> EmotionVector | None:
    """Return the curated ``EmotionVector`` for a preset key, or ``None``.

    Unlike :func:`build_emotion_vector`, this does **not** substitute a default
    for unknown labels — it returns ``None`` so callers can distinguish "this
    was a recognised mood preset" from "no match" (e.g. the refinement
    executor's free-form ``emotion`` setter, which must leave the existing
    vector untouched on a miss rather than clobber it with the default).
    """
    return _EMOTION_PRESETS.get(mood)


def build_emotion_vector(mood: str) -> EmotionVector:
    """Map a coarse mood-preset key to a curated ``EmotionVector``.

    Used by the mood-responder fallback (when GPT is down). Unknown keys fall
    back to a mildly-positive default — callers that need to treat an unknown
    key as "no signal" should use :func:`lookup_emotion_preset` instead.
    """
    return _EMOTION_PRESETS.get(mood, _DEFAULT_EMOTION_VECTOR)


def normalize_mood_label(label: str | None) -> str | None:
    """Normalise a mood label to a canonical preset key, or ``None``.

    Applies known aliases (e.g. ``melancholy`` -> ``melancholic``); returns the
    label unchanged if it is already a canonical preset; otherwise ``None`` so
    callers can fall back (e.g. to ``neutral`` in the UI dropdown).
    """
    if not label:
        return None
    canonical = _MOOD_LABEL_ALIASES.get(label, label)
    return canonical if canonical in _EMOTION_PRESETS else None


class MelodySuggestion(SchemaModel):
    midi_bytes: bytes = Field(...)
    notes: list[NoteEvent] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    coherence_score: float = Field(..., ge=0, le=1)
    engine: str = "mock"
    model_id: str | None = None
    avg_log_prob: float | None = None
    constraint_trace: dict[str, Any] = Field(default_factory=dict)
    score_breakdown: dict[str, Any] = Field(default_factory=dict)


class LyricSuggestion(SchemaModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    syllable_count: int = Field(..., ge=0)


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


class ChordKeyConstraint(SchemaModel):
    """Inference-time soft key filter applied to the pitch-class Q head.

    Records which PCs were demoted (out-of-scale relative to the declared key)
    and the per-PC Q delta so explanations can reconstruct the filter's effect.
    """

    applied: bool
    declared_key: str | None = None
    scale_pcs: list[int] = Field(default_factory=list)
    out_of_scale_pcs: list[int] = Field(default_factory=list)
    lambda_key: float = 0.0
    q_delta_per_pc: list[float] = Field(default_factory=list)
    rationale: str = ""


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
    key_constraint: ChordKeyConstraint | None = None
    selected: SelectedChord
    model_kind: Literal["dqn"] = "dqn"
    noise_state: Literal["disabled", "enabled"] = "disabled"
    noise_samples: int | None = None

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
    chords: list[str] = Field(
        default_factory=list,
        description=(
            "Chord symbols in playback order. Convention: one chord per bar "
            "(4 beats in 4/4). The Phase 5 chord-consonance constraint in "
            "ml/melody_sketchpad/continuation/constraints.py depends on this — "
            "if you change a producer (DQN, lyric templates, user input, mock) "
            "to emit variable durations, update the chord-index math there too."
        ),
    )
    score: float = Field(..., ge=0, le=1)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    mood_alignment: float | None = Field(default=None, ge=0, le=1)
    harmonic_function: str | None = None
    explanation: str = Field(..., min_length=1)
    native_distributions: list[ChordDistribution] = Field(default_factory=list)
    chord_annotations: list[ChordAnnotation] = Field(default_factory=list)
    chord_explanations: list[ChordExplanation] = Field(default_factory=list)


class RefinementOp(SchemaModel):
    target: Literal["harmonizer", "melody_generator", "lyric_generator", "session"]
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_params_for_target(self):
        _validate_refinement_params(self.target, self.params)
        return self


class RefinementPlan(SchemaModel):
    operations: list[RefinementOp] = Field(..., min_length=1)
    interpretation: str = Field(..., min_length=1)


_REFINEMENT_PARAM_RULES: dict[str, dict[str, Any]] = {
    "harmonizer": {
        "requested_target": "str",
        "chord_complexity": {"low", "medium", "high"},
        "extensions": "list[str]",
        "emotion_valence": (-1.0, 1.0),
        "emotion_valence_delta": (-1.0, 1.0),
        "prefer_minor_color": "bool",
        "reduce_minor_bias": "bool",
        "section": "str",
        "tension": {"lower", "low", "medium", "higher", "high"},
    },
    "melody_generator": {
        "requested_target": "str",
        "contour": {"rising", "falling", "arch", "valley", "varied"},
        "rhythmic_density": {"lighter", "steady", "denser"},
        "length": {"shorter", "longer", "unchanged"},
        "section": "str",
        "smooth_contour": "bool",
        "register_shift": {"slightly_up", "slightly_down", "up", "down", "none"},
    },
    "lyric_generator": {
        "requested_target": "str",
        "imagery": "str",
        "num_options": (1, 5),
        "preserve_syllable_targets": "bool",
        "section": "str",
        "language": {"en", "ru"},
        "length": {"shorter", "longer", "unchanged"},
        "preserve_phrase": "str",
    },
    "session": {
        "requested_target": "str",
        "preserve": "list[str]",
        "emotion": "str",
        "emotion_valence": (-1.0, 1.0),
        "emotion_arousal": (-1.0, 1.0),
        "genre": "str",
    },
}


def _validate_refinement_params(target: str, params: dict[str, Any]) -> None:
    rules = _REFINEMENT_PARAM_RULES[target]
    unknown = sorted(set(params) - set(rules))
    if unknown:
        raise ValueError(f"Unknown params for {target}: {', '.join(unknown)}")
    for name, value in params.items():
        _validate_refinement_param(target, name, value, rules[name])


def _validate_refinement_param(target: str, name: str, value: Any, rule: Any) -> None:
    if rule == "str":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{target}.{name} must be a non-empty string")
        return
    if rule == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{target}.{name} must be boolean")
        return
    if rule == "list[str]":
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"{target}.{name} must be a list of non-empty strings")
        return
    if isinstance(rule, set):
        if value not in rule:
            choices = ", ".join(sorted(str(item) for item in rule))
            raise ValueError(f"{target}.{name} must be one of: {choices}")
        return
    if isinstance(rule, tuple):
        lower, upper = rule
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < lower or value > upper:
            raise ValueError(f"{target}.{name} must be between {lower} and {upper}")
        return
    raise ValueError(f"Unsupported refinement param rule for {target}.{name}")


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
    # The human-readable mood label behind ``emotion_vector`` and its
    # provenance. ``emotion_source`` lets generation decide whether to respect
    # an author's Song Brief choice ("authored") or treat the current vector as
    # an auto-inferred guess that may be re-suggested ("detected").
    mood_label: str | None = None
    emotion_source: Literal["authored", "detected"] | None = None
    melody_suggestions: list[MelodySuggestion] = Field(default_factory=list)
    lyric_suggestions: list[LyricSuggestion] = Field(default_factory=list)
    explanation_report: ExplanationReport | None = None
    chat_history: list[ChatMessage] = Field(default_factory=list)
    user_params: dict[str, Any] = Field(default_factory=dict)
    history: list[Action] = Field(default_factory=list)
    # Optional song-form labels. When both are set, the continuation
    # pipeline switches to BiMMuDa-conditional constraints (e.g. verse
    # primer + chorus target -> expect register lift, slight density drop).
    # When either is None, the pipeline falls back to primer-relative
    # constraints (continue stylistically near the primer).
    primer_section: Section | None = None
    target_section: Section | None = None


class SessionSummary(SchemaModel):
    session_id: UUID
    created_at: str
    updated_at: str
    pipeline_version: str = Field(..., min_length=1)


Progression = ChordProgression
