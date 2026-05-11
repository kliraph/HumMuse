"""Emotion-to-temperature scheduling for melody continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ml.melody_sketchpad.continuation.inference import EnsembleMember, EnsembleMelodyContinuationModel
from shared.schemas import EmotionVector

_DEFAULT_CENTER = 0.85
_MIN_TEMPERATURE = 0.40
_MAX_TEMPERATURE = 1.20
_MIN_WIDTH_SCALE = 0.70
_MAX_WIDTH_SCALE = 1.5421052631578946


def emotion_temperature_schedule(
    emotion_vector: EmotionVector | dict[str, float] | None,
    *,
    slots: int = 3,
) -> tuple[float, ...]:
    """
    Map arousal to a sampling-temperature schedule.

    Higher arousal widens the schedule, yielding more diverse sampling. Lower
    arousal narrows it around a conservative center. Valence is intentionally
    ignored here; it belongs to scoring/selection, not raw sampling diversity.
    """
    if slots <= 0:
        raise ValueError("slots must be positive")
    if slots == 1:
        return (_DEFAULT_CENTER,)

    arousal = _emotion_arousal(emotion_vector)
    arousal_unit = (arousal + 1.0) / 2.0
    half_width = 0.06 + (0.29 * arousal_unit)
    start = _clamp(_DEFAULT_CENTER - half_width, _MIN_TEMPERATURE, _MAX_TEMPERATURE)
    stop = _clamp(_DEFAULT_CENTER + half_width, _MIN_TEMPERATURE, _MAX_TEMPERATURE)
    step = (stop - start) / (slots - 1)
    return tuple(round(start + (step * index), 3) for index in range(slots))


def rescale_temperature_schedule(
    base_schedule: Sequence[float],
    emotion_vector: EmotionVector | dict[str, float] | None,
) -> tuple[float, ...]:
    """
    Widen or narrow a member's own temperature schedule based on arousal.

    The schedule center is preserved so checkpoint/model personality remains
    orthogonal to emotion. Arousal only changes sampling width.
    """
    if not base_schedule:
        raise ValueError("base_schedule must be non-empty")
    if len(base_schedule) == 1:
        return (round(float(base_schedule[0]), 3),)

    values = tuple(float(value) for value in base_schedule)
    center = sum(values) / len(values)
    scale = _width_scale(emotion_vector)
    return tuple(
        round(_clamp(center + ((value - center) * scale), _MIN_TEMPERATURE, _MAX_TEMPERATURE), 3)
        for value in values
    )


def generate_ensemble_with_emotion_temperature(
    ensemble: EnsembleMelodyContinuationModel,
    prompt_midi_path: Path | str,
    *,
    emotion_vector: EmotionVector | dict[str, float] | None,
    max_new_tokens: int = 128,
    out_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate from an ensemble after rescaling each member's base schedule.

    Member counts and declaration order are preserved; only the per-member
    temperatures passed to generation are changed.
    """
    original_members = ensemble.members
    try:
        ensemble.members = [
            EnsembleMember(
                checkpoint_path=member.checkpoint_path,
                tokenizer_path=member.tokenizer_path,
                model_id=member.model_id,
                temperatures=rescale_temperature_schedule(member.temperatures, emotion_vector),
                count=member.count,
            )
            for member in original_members
        ]
        return ensemble.generate_candidates(
            prompt_midi_path,
            max_new_tokens=max_new_tokens,
            out_dir=out_dir,
        )
    finally:
        ensemble.members = original_members


def _width_scale(emotion_vector: EmotionVector | dict[str, float] | None) -> float:
    arousal_unit = (_emotion_arousal(emotion_vector) + 1.0) / 2.0
    return _MIN_WIDTH_SCALE + ((_MAX_WIDTH_SCALE - _MIN_WIDTH_SCALE) * arousal_unit)


def _emotion_arousal(emotion_vector: EmotionVector | dict[str, float] | None) -> float:
    if emotion_vector is None:
        return 0.0
    if isinstance(emotion_vector, EmotionVector):
        return _clamp(float(emotion_vector.arousal), -1.0, 1.0)
    return _clamp(float(emotion_vector.get("arousal", 0.0)), -1.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
