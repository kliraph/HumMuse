"""Mode-aware emotion-conditioned additive Q bias for the DQN pitch-class head.

The valence bias targets depend on the declared key's mode. In a minor key
with positive valence we never suppress the minor third (♭3); we boost the
tonic minor triad and add Dorian/harmonic-minor brightening tones (raised 6,
raised 7). This avoids the failure mode where "bright in C minor" silently
rewrote chord choices toward chord vocabularies that exclude E♭. Arousal
keeps the independent Lerdahl tension/release path.
"""

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
    """Return pitch-class Q-values with a mode-aware emotion bias added.

    Valence moves Q toward mode-appropriate targets (see :func:`emotion_bias_vector`).
    Arousal adds an independent Lerdahl-based key-relative tension/release bias;
    neither path renormalizes probabilities.
    """

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = emotion_bias_vector(q_tensor, valence=valence, arousal=arousal, key=key)
    return q_tensor + bias


def emotion_bias_vector(q_pc: Any, valence: float, arousal: float, key: str | None = "C major") -> Any:
    """Build the additive 13-slot Q bias vector without applying it.

    Valence targets depend on declared mode:

    * Major key, +valence: boost the major 3rd, suppress the minor 3rd. (Major
      and parallel-minor triads share root+fifth, so the third is the only
      differentiating tone — moving just it is the same as moving both triads
      with the shared tones cancelled.)
    * Major key, −valence: boost the minor 3rd, suppress the major 3rd
      (modal mixture).
    * Minor key, +valence: boost tonic minor triad **and** raised 6 + raised
      7 (Dorian / harmonic-minor brightening); never suppresses the ♭3.
    * Minor key, −valence: boost tonic minor triad **and** ♭6
      (Phrygian / Aeolian darkening); never suppresses the ♭3.

    Arousal then adds the Lerdahl tension/release bias on top.
    """

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = torch.zeros_like(q_tensor)
    flat_bias = bias.view(-1)
    tonic_pc, mode = parse_key(key)
    valence_clipped = _clip(valence, -1.0, 1.0)
    boost, suppress = _mode_aware_targets(tonic_pc=tonic_pc, mode=mode, valence=valence_clipped)
    magnitude = _bias_strength(valence_clipped, arousal) * abs(valence_clipped)
    for pc in boost:
        flat_bias[pc] += magnitude
    for pc in suppress:
        flat_bias[pc] -= magnitude
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
    tonic_pc, mode = parse_key(key)
    valence = _clip(valence, -1.0, 1.0)
    arousal = _clip(arousal, -1.0, 1.0)
    if abs(valence) < 1e-12 and abs(arousal) < 1e-12:
        return "No bias applied (valence ~= 0 and arousal ~= 0)."
    if abs(valence) < 1e-12:
        direction = "tension" if arousal > 0 else "release"
        return (
            f"Applied arousal {direction} Q bias around tonic pitch class {tonic_pc} ({mode} key); "
            f"tension strength={0.2 * abs(arousal):.3f}."
        )
    descriptor = _mode_aware_descriptor(mode=mode, valence=valence)
    return (
        f"Applied {descriptor} around tonic pitch class {tonic_pc} ({mode} key); "
        f"arousal scaled bias strength to {_bias_strength(valence, arousal):.3f} "
        "and added Lerdahl tension/release bias."
    )


def lerdahl_tension(key: str | None = "C major") -> list[float]:
    tonic, mode = parse_key(key)
    profile = LERDAHL_MINOR if mode == "minor" else LERDAHL_MAJOR
    mean = sum(profile) / 12
    return [mean - profile[(pc - tonic) % 12] for pc in range(12)]


def _mode_aware_targets(*, tonic_pc: int, mode: str, valence: float) -> tuple[set[int], set[int]]:
    """Return (boost_pcs, suppress_pcs) for a given (mode, valence-sign) pair.

    Major key: the major triad and the parallel-minor triad share the root and
    fifth, so the *only* differentiating tone is the third. We therefore move
    just the third — boost the major 3rd / suppress the minor 3rd for positive
    valence, reversed for negative (modal mixture). (Targeting both full triads
    would add and subtract the shared root+fifth, cancelling them to zero; this
    is the same vector expressed honestly, so the recorded q_delta and the
    rationale match what actually changes.)

    Minor key: branches deliberately leave ``suppress`` empty so the ♭3 of the
    declared key is never demoted — the failure mode that motivated the rewrite.
    """

    if mode == "major":
        major_third = {(tonic_pc + 4) % 12}
        minor_third = {(tonic_pc + 3) % 12}
        if valence >= 0:
            return major_third, minor_third
        return minor_third, major_third

    # mode == "minor"
    minor_triad = {(tonic_pc + i) % 12 for i in (0, 3, 7)}
    if valence >= 0:
        brightening = {(tonic_pc + 9) % 12, (tonic_pc + 11) % 12}
        return minor_triad | brightening, set()
    darkening = {(tonic_pc + 8) % 12}
    return minor_triad | darkening, set()


def _mode_aware_descriptor(*, mode: str, valence: float) -> str:
    if mode == "major":
        return (
            "major-third Q bias (boosting the major 3rd, suppressing the minor 3rd)"
            if valence >= 0
            else "minor-third Q bias (boosting the minor 3rd, suppressing the major 3rd)"
        )
    if valence >= 0:
        return "tonic minor-triad Q bias plus Dorian/harmonic-minor brightening tones"
    return "tonic minor-triad Q bias plus Phrygian/Aeolian ♭6 darkening color"


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
