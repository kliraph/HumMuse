"""Execute a parsed :class:`RefinementPlan` against the live session modules.

Each ``RefinementOp`` is dispatched to a target-specific handler:

* ``harmonizer``        -> adjust ``state.emotion_vector`` and re-run the DQN
                          chord generator; ``state.chord_progressions`` is
                          replaced.
* ``melody_generator``  -> record shaping params; re-run the Music Transformer
                          continuation pipeline; ``state.melody_suggestions``
                          is replaced.
* ``lyric_generator``   -> call the GPT lyric pipeline with the op params;
                          ``state.lyric_suggestions`` is replaced.
* ``session``           -> merge global mood / genre / preserve hints into
                          ``state.user_params``; no regeneration.

Handlers degrade gracefully: missing prerequisites (no melody, no GPT
pipeline) produce ``status="skipped"`` with a human-readable reason rather
than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from backend.logging_config import get_logger
from backend.lyric_responder import generate_lyric_suggestions
from backend.mock_pipeline import build_emotion_vector
from ml.gpt.pipeline import GPTPipeline
from ml.harmony import generate_chords as generate_dqn_chords
from ml.melody_sketchpad.continuation.pipeline import continue_melody as run_continuation_pipeline
from shared.schemas import (
    EmotionVector,
    RefinementOp,
    RefinementPlan,
    SessionState,
)

LOGGER = get_logger("backend.refinement_executor")

OpStatus = Literal["applied", "skipped", "failed"]

_VALENCE_BOUNDS = (-1.0, 1.0)
_AROUSAL_BOUNDS = (-1.0, 1.0)
_TENSION_VALENCE_DELTA: dict[str, float] = {
    "lower": 0.25,
    "low": 0.25,
    "higher": -0.25,
    "high": -0.25,
}
_LENGTH_MAX_NEW_TOKENS: dict[str, int] = {
    "shorter": 64,
    "longer": 192,
    "unchanged": 128,
}


@dataclass
class OperationResult:
    """Outcome of executing one :class:`RefinementOp`."""

    target: str
    status: OpStatus
    rationale: str
    changes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "status": self.status,
            "rationale": self.rationale,
            "changes": self.changes,
            "error": self.error,
        }


@dataclass
class RefinementExecution:
    """Aggregate of all operation results from a single plan."""

    results: list[OperationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"results": [result.to_dict() for result in self.results]}

    @property
    def applied_count(self) -> int:
        return sum(1 for result in self.results if result.status == "applied")

    @property
    def any_applied(self) -> bool:
        return self.applied_count > 0


def execute_refinement_plan(
    state: SessionState,
    plan: RefinementPlan,
    *,
    pipeline: GPTPipeline | None = None,
) -> RefinementExecution:
    """Execute ``plan.operations`` in order, mutating ``state`` in place."""
    execution = RefinementExecution()
    for op in plan.operations:
        try:
            result = _dispatch(state, op, pipeline=pipeline)
        except Exception as exc:  # never let one op kill the rest
            LOGGER.info(
                "refinement_op_failed",
                target=op.target,
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            result = OperationResult(
                target=op.target,
                status="failed",
                rationale=op.rationale,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        execution.results.append(result)
    return execution


def _dispatch(
    state: SessionState,
    op: RefinementOp,
    *,
    pipeline: GPTPipeline | None,
) -> OperationResult:
    if op.target == "harmonizer":
        return _execute_harmonizer(state, op)
    if op.target == "melody_generator":
        return _execute_melody(state, op)
    if op.target == "lyric_generator":
        return _execute_lyric(state, op, pipeline=pipeline)
    if op.target == "session":
        return _execute_session(state, op)
    return OperationResult(
        target=op.target,
        status="failed",
        rationale=op.rationale,
        error=f"Unknown refinement target: {op.target}",
    )


# --- harmonizer ---------------------------------------------------------------


def _execute_harmonizer(state: SessionState, op: RefinementOp) -> OperationResult:
    params = op.params
    if not state.melody_notes:
        return OperationResult(
            target=op.target,
            status="skipped",
            rationale=op.rationale,
            error="no_melody",
            changes={"reason": "DQN harmonizer needs melody notes; none on session."},
        )

    before = state.emotion_vector or EmotionVector(valence=0.0, arousal=0.3)
    new_vector = _adjust_emotion_vector(before, params)
    state.emotion_vector = new_vector

    progressions = generate_dqn_chords(
        state.melody_notes,
        key=state.detected_key or "C major",
        emotion_vector=new_vector,
        top_k=3,
        tempo_bpm=state.detected_tempo or 120.0,
    )
    state.chord_progressions = progressions

    unsupported = _collect_unsupported_harmonizer_params(params)
    changes: dict[str, Any] = {
        "emotion_vector_before": before.model_dump(),
        "emotion_vector_after": new_vector.model_dump(),
        "progression_count": len(progressions),
        "top_progression": progressions[0].chords if progressions else None,
    }
    if unsupported:
        changes["unsupported_params"] = unsupported
    return OperationResult(
        target=op.target,
        status="applied",
        rationale=op.rationale,
        changes=changes,
    )


def _adjust_emotion_vector(current: EmotionVector, params: dict[str, Any]) -> EmotionVector:
    valence = current.valence
    arousal = current.arousal

    if "emotion_valence" in params:
        valence = float(params["emotion_valence"])
    elif "emotion_valence_delta" in params:
        valence += float(params["emotion_valence_delta"])

    tension = params.get("tension")
    if isinstance(tension, str):
        delta = _TENSION_VALENCE_DELTA.get(tension.lower())
        if delta is not None:
            valence += delta
            # Tension also nudges arousal: higher tension -> higher arousal.
            arousal += -delta if delta < 0 else -delta
            arousal = _clip(arousal + (0.15 if delta < 0 else -0.15), *_AROUSAL_BOUNDS)

    if params.get("prefer_minor_color") is True:
        valence = min(valence, -0.2)
    if params.get("reduce_minor_bias") is True:
        valence = max(valence, 0.2)

    return EmotionVector(
        valence=_clip(valence, *_VALENCE_BOUNDS),
        arousal=_clip(arousal, *_AROUSAL_BOUNDS),
    )


def _collect_unsupported_harmonizer_params(params: dict[str, Any]) -> list[str]:
    """Params the DQN cannot act on yet, recorded for downstream context."""
    keys = []
    for key in ("chord_complexity", "extensions", "section"):
        if key in params:
            keys.append(key)
    return keys


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --- melody_generator ---------------------------------------------------------


def _execute_melody(state: SessionState, op: RefinementOp) -> OperationResult:
    params = op.params
    if not state.melody_notes:
        return OperationResult(
            target=op.target,
            status="skipped",
            rationale=op.rationale,
            error="no_melody",
            changes={"reason": "Continuation needs a primer melody; none on session."},
        )

    length = params.get("length")
    max_new_tokens = _LENGTH_MAX_NEW_TOKENS.get(length, _LENGTH_MAX_NEW_TOKENS["unchanged"])
    state.user_params["last_melody_shaping"] = dict(params)

    suggestions = run_continuation_pipeline(
        state,
        top_n=3,
        max_new_tokens=max_new_tokens,
    )
    state.melody_suggestions = suggestions

    unsupported = _collect_unsupported_melody_params(params)
    changes: dict[str, Any] = {
        "shaping_params": dict(params),
        "max_new_tokens": max_new_tokens,
        "suggestion_count": len(suggestions),
    }
    if unsupported:
        changes["unsupported_params"] = unsupported
    return OperationResult(
        target=op.target,
        status="applied",
        rationale=op.rationale,
        changes=changes,
    )


def _collect_unsupported_melody_params(params: dict[str, Any]) -> list[str]:
    keys = []
    for key in ("contour", "rhythmic_density", "section", "smooth_contour", "register_shift"):
        if key in params:
            keys.append(key)
    return keys


# --- lyric_generator ----------------------------------------------------------


def _execute_lyric(
    state: SessionState,
    op: RefinementOp,
    *,
    pipeline: GPTPipeline | None,
) -> OperationResult:
    """Run the lyric_generator op through the shared lyric responder.

    The responder handles GPT-first with mock fallback (and logs transport
    failures via the ``lyric_suggest_gpt_failed`` event at INFO, which the
    structlog config surfaces to stdout while the API is running). The op is
    always ``applied`` because the responder always produces *some*
    suggestions — what the user sees in ``changes`` is whether they came from
    GPT or the mock and, if mock, why.
    """
    params = op.params
    instruction = _build_lyric_instruction(params)
    num_suggestions = int(params.get("num_options", 3))

    result = generate_lyric_suggestions(
        state,
        mode="Poetic",
        num_suggestions=num_suggestions,
        pipeline=pipeline,
        extra_instruction=instruction,
    )
    state.lyric_suggestions = result.suggestions
    return OperationResult(
        target=op.target,
        status="applied",
        rationale=op.rationale,
        changes={
            "suggestion_count": len(result.suggestions),
            "source": result.source,
            "mode": result.mode,
            "model_version": result.model_version,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
            "instruction": instruction,
        },
        error=result.error,
    )


def _build_lyric_instruction(params: dict[str, Any]) -> str:
    """Compose a free-form instruction string from structured lyric params."""
    parts: list[str] = []
    if imagery := params.get("imagery"):
        parts.append(f"Use this imagery: {imagery}.")
    if preserve := params.get("preserve_phrase"):
        parts.append(f"Preserve this phrase: \"{preserve}\".")
    length = params.get("length")
    if length in ("shorter", "longer"):
        parts.append(f"Make the line {length}.")
    if params.get("preserve_syllable_targets") is True:
        parts.append("Match the existing syllable targets exactly.")
    if (lang := params.get("language")) == "ru":
        parts.append("Write the lyrics in Russian.")
    elif lang == "en":
        parts.append("Write the lyrics in English.")
    if section := params.get("section"):
        parts.append(f"Target section: {section}.")
    if not parts:
        parts.append("Generate fresh lyric options that fit the current session context.")
    return " ".join(parts)


# --- session ------------------------------------------------------------------


def _execute_session(state: SessionState, op: RefinementOp) -> OperationResult:
    params = op.params
    merged_keys: list[str] = []

    if "preserve" in params:
        state.user_params["preserve"] = list(params["preserve"])
        merged_keys.append("preserve")
    if "genre" in params:
        state.user_params["genre"] = params["genre"]
        merged_keys.append("genre")
    if "mood" in params:
        state.user_params["mood"] = params["mood"]
        merged_keys.append("mood")
        state.emotion_vector = build_emotion_vector(str(params["mood"]))
    if "emotion" in params:
        state.user_params["emotion"] = params["emotion"]
        merged_keys.append("emotion")
        # `emotion` is a free-form label from the parser; map to a vector when
        # it happens to match a preset. Otherwise leave the existing vector
        # alone — the harmonizer ops are the canonical valence/arousal lever.
        state.emotion_vector = build_emotion_vector(str(params["emotion"]))

    return OperationResult(
        target=op.target,
        status="applied",
        rationale=op.rationale,
        changes={
            "merged_keys": merged_keys,
            "emotion_vector": state.emotion_vector.model_dump() if state.emotion_vector else None,
        },
    )
