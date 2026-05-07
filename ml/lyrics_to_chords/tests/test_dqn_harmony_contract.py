"""Lightweight DQN harmony contract checks used by lyrics-to-chords flows."""

from __future__ import annotations

import torch

from ml.harmony.emotion_modulation import emotion_q_bias
from ml.harmony.reward_attribution import compute_attribution


def test_minor_valence_bias_and_attribution_contract_are_available_to_lyrics_flow() -> None:
    q_pc = torch.zeros(13)
    q_pc[4] = 0.4

    biased = emotion_q_bias(q_pc, valence=-1.0, arousal=0.0, key="C major")
    attribution = compute_attribution(
        {"is_rest": False, "octave": 3, "inversion": 0, "pcs": [0, 3, 7], "pc_names": ["C", "D#", "G"]},
        (63, 2, 0),
        None,
        "C major",
    )

    assert int(torch.argmax(biased).item()) == 3
    assert set(attribution) == {
        "harmony_rule",
        "progression_penalty",
        "chord_tone_inclusion",
        "mutual_info",
        "key_fit_proxy",
    }
