"""Emotion-conditioned additive Q bias for DQN pitch-class heads."""

from __future__ import annotations

from typing import Any

from ml.harmony.chord_symbols import parse_key
from shared.schemas import EmotionVector

try:
    import torch
except ImportError:  # pragma: no cover - exercised only on non-ML installs.
    torch = None

LERDAHL_MAJOR = [5, 1, 2, 1, 3, 2, 1, 4, 1, 2, 1, 2]
LERDAHL_MINOR = [5, 1, 2, 3, 1, 2, 1, 4, 2, 1, 2, 1]


def emotion_q_bias(q_pc: Any, valence: float, arousal: float, key: str | None = "C major") -> Any:
    """Return pitch-class Q-values with a key-relative emotion bias added.

    Valence moves Q evidence toward tonic major/minor thirds. Arousal adds an
    independent key-relative tension/release bias; neither path renormalizes
    probabilities.
    """

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = emotion_bias_vector(q_tensor, valence=valence, arousal=arousal, key=key)
    return q_tensor + bias


def emotion_bias_vector(q_pc: Any, valence: float, arousal: float, key: str | None = "C major") -> Any:
    """Build the additive 13-slot Q bias vector without applying it."""

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = torch.zeros_like(q_tensor)
    flat_bias = bias.view(-1)
    tonic_pc, _ = parse_key(key)
    major_pcs = {(tonic_pc + interval) % 12 for interval in (0, 4, 7)}
    minor_pcs = {(tonic_pc + interval) % 12 for interval in (0, 3, 7)}
    strength = _bias_strength(valence, arousal)
    for pc in major_pcs:
        flat_bias[pc] += strength * _clip(valence, -1.0, 1.0)
    for pc in minor_pcs:
        flat_bias[pc] -= strength * _clip(valence, -1.0, 1.0)
    bias = bias + _arousal_tension_bias(q_tensor, arousal, key)
    return bias


def emotion_values(emotion_vector: EmotionVector | dict[str, float] | None) -> tuple[float, float] | None:
    if emotion_vector is None:
        return None
    if isinstance(emotion_vector, EmotionVector):
        return (
            _clip(float(emotion_vector.valence), -1.0, 1.0),
            _clip(float(emotion_vector.arousal), -1.0, 1.0),
        )
    return (
        _clip(float(emotion_vector.get("valence", 0.0)), -1.0, 1.0),
        _clip(float(emotion_vector.get("arousal", 0.0)), -1.0, 1.0),
    )


def emotion_bias_rationale(valence: float, arousal: float, key: str | None = "C major") -> str:
    tonic, _ = parse_key(key)
    if abs(valence) < 1e-12 and abs(arousal) < 1e-12:
        return "No bias applied (valence ~= 0 and arousal ~= 0)."
    if abs(valence) < 1e-12:
        direction = "tension" if arousal > 0 else "release"
        return (
            f"Applied arousal {direction} Q bias around tonic pitch class {tonic}; "
            f"arousal tension strength={0.2 * abs(_clip(arousal, -1.0, 1.0)):.3f}."
        )
    mood_direction = "major-triad" if valence > 0 else "minor-triad"
    return (
        f"Applied {mood_direction} Q bias around tonic pitch class {tonic}; "
        f"arousal scaled bias strength to {_bias_strength(valence, arousal):.3f} "
        "and added key-relative tension/release bias."
    )


def lerdahl_tension(key: str | None = "C major") -> list[float]:
    tonic, mode = parse_key(key)
    profile = LERDAHL_MINOR if mode == "minor" else LERDAHL_MAJOR
    mean = sum(profile) / 12
    return [mean - profile[(pc - tonic) % 12] for pc in range(12)]


def _arousal_tension_bias(q_pc: Any, arousal: float, key: str | None, *, lam: float = 0.2) -> Any:
    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = torch.zeros_like(q_tensor)
    flat = bias.view(-1)
    tension = lerdahl_tension(key)
    a = _clip(arousal, -1.0, 1.0)
    for pc in range(12):
        flat[pc] += lam * a * tension[pc]
    return bias


def _bias_strength(valence: float, arousal: float) -> float:
    base = 0.85
    arousal_scale = 1.0 + (0.5 * _clip(arousal, -1.0, 1.0))
    return base * arousal_scale


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("DQN emotion Q bias requires torch to be installed.")
