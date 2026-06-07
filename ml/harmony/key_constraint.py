"""Soft inference-time key filter for DQN pitch-class Q-values (Road B).

The trained RL-Chord DQN does not receive a key signal in its input — it
infers tonal centre from the local melody window. For chromatic-leaning or
short melodies this is unreliable and the model can lock onto an off-key
chord vocabulary. This module subtracts ``lambda_key`` from Q-values of
pitch classes that are *not* in the declared key's scale, leaving the
chromatic option open only when the model is confident enough to overcome
the penalty.

This is a deliberately soft filter, not a hard mask. It preserves the
ability to render secondary dominants and modal-mixture chords when the
trained policy genuinely prefers them, while pushing back against the
silent off-key drift observed in real sessions.
"""

from __future__ import annotations

from typing import Any

from ml.harmony.chord_symbols import parse_key

try:
    import torch
except ImportError:  # pragma: no cover - exercised only on non-ML installs.
    torch = None

# Natural-major and natural-minor scale intervals relative to the tonic.
_SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}

DEFAULT_LAMBDA_KEY = 1.5


def key_scale_pcs(key: str | None) -> set[int] | None:
    """Return the set of pitch classes in the declared key's natural scale.

    Returns ``None`` when no key is supplied so callers can treat that as
    "no filter applicable".
    """

    if not key:
        return None
    tonic_pc, mode = parse_key(key)
    intervals = _SCALE_INTERVALS.get(mode, _SCALE_INTERVALS["major"])
    return {(tonic_pc + interval) % 12 for interval in intervals}


def key_filter_bias(q_pc: Any, key: str | None, lambda_key: float = DEFAULT_LAMBDA_KEY) -> Any:
    """Return the additive 13-slot bias that demotes out-of-scale PCs.

    PC 12 (the triad-vs-tetrad sentinel) is always left untouched.
    """

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    bias = torch.zeros_like(q_tensor)
    scale = key_scale_pcs(key)
    if scale is None or lambda_key == 0.0:
        return bias
    flat = bias.view(-1)
    for pc in range(12):
        if pc not in scale:
            flat[pc] -= float(lambda_key)
    return bias


def apply_key_filter(q_pc: Any, key: str | None, lambda_key: float = DEFAULT_LAMBDA_KEY) -> Any:
    """Apply the soft key filter and return the biased Q tensor."""

    _require_torch()
    q_tensor = torch.as_tensor(q_pc)
    return q_tensor + key_filter_bias(q_tensor, key, lambda_key)


def out_of_scale_pcs(key: str | None) -> list[int]:
    scale = key_scale_pcs(key)
    if scale is None:
        return []
    return sorted(pc for pc in range(12) if pc not in scale)


def key_filter_rationale(key: str | None, lambda_key: float) -> str:
    if not key:
        return "No declared key; key filter inactive."
    if lambda_key == 0.0:
        return f"Key filter for {key} configured but lambda_key=0; inactive."
    scale = key_scale_pcs(key) or set()
    out = out_of_scale_pcs(key)
    return (
        f"Soft key filter applied for {key}: scale PCs {sorted(scale)}, "
        f"demoted out-of-scale PCs {out} by -{lambda_key:.2f}."
    )


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("The key filter requires torch to be installed.")
