"""Tests for deterministic chord explanations."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.harmony.chord_symbols import derive_chord_symbol
from ml.harmony.dqn import DEFAULT_CHECKPOINT, generate_chords
from ml.harmony.explain import build_explanation, prose_template
from shared.schemas import ChordDistribution, EmotionVector, NoteEvent

PC_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel"]


def _distribution() -> ChordDistribution:
    membership = [0.01] * 13
    for pc in (0, 4, 7):
        membership[pc] = 0.2
    membership[12] = 0.3
    total = sum(membership)
    membership = [value / total for value in membership]
    return ChordDistribution(
        position=0,
        policy={
            "rest": {"rest": 0.01, "chord": 0.99},
            "octave": {"2": 0.2, "3": 0.8},
            "inversion": {"0": 0.7, "1": 0.1, "2": 0.1, "3": 0.1},
            "pitch_class": {label: membership[index] for index, label in enumerate(PC_LABELS[:12])},
            "pitch_class_membership": membership,
            "pitch_class_labels": PC_LABELS,
        },
        q_values={
            "rest": -2.0,
            "octave": [0.1, 0.8],
            "inversion": [1.2, 0.3, 0.2, 0.1],
            "pitch_class": [4.0, 0.1, 0.1, 0.1, 3.5, 0.1, 0.1, 3.0, 0.1, 0.1, 0.1, 0.1, 2.8],
        },
        q_margin={"rest": 2.0, "octave": 0.7, "inversion": 0.9, "pitch_class": 0.5},
        dueling={
            "value_rest": -2.0,
            "value_octave": 0.4,
            "value_inversion": 0.5,
            "value_pc": 2.0,
            "advantage_rest": [0.0],
            "advantage_octave": [-0.2, 0.2],
            "advantage_inversion": [0.4, -0.1, -0.1, -0.2],
            "advantage_pc": [1.0, -0.2, -0.2, -0.2, 0.8, -0.2, -0.2, 0.6, -0.2, -0.2, -0.2, -0.2, 0.4],
        },
        context={
            "prev_chord_pcs": [7, 11, 2],
            "prev_chord_symbol": "G",
            "current_note": {"pitch": 60, "duration": 2, "position": 0},
        },
        reward_attribution={
            "harmony_rule": 12.0,
            "progression_penalty": 0.0,
            "chord_tone_inclusion": 1.0,
            "mutual_info": 0.0,
            "key_fit_proxy": 1.0,
        },
        emotion_bias={
            "applied": True,
            "q_delta_per_pc": [0.0, 0.0, 0.0, -0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "rationale": "Applied major-triad Q bias around tonic pitch class 0.",
        },
        selected={"is_rest": False, "octave": 3, "inversion": 0, "pcs": [0, 4, 7], "pc_names": ["C", "E", "G"]},
    )


def test_build_explanation_pulls_q_dueling_context_rewards_and_emotion() -> None:
    distribution = _distribution()
    derivation = derive_chord_symbol(distribution, key="C major")

    explanation = build_explanation(distribution, derivation)

    assert explanation.chord_symbol == "C"
    assert explanation.prev_chord_link == "G"
    assert explanation.margin == 0.5
    assert explanation.value_score == 2.0
    assert explanation.reward_components["harmony_rule"] == 12.0
    assert explanation.emotion_bias_summary is not None
    assert "Q=" in explanation.prose


def test_prose_template_is_deterministic() -> None:
    distribution = _distribution()
    explanation = build_explanation(distribution, derive_chord_symbol(distribution, key="C major"))

    assert prose_template(explanation) == prose_template(explanation)
    assert "over" in explanation.prose
    assert "Rewards:" in explanation.prose


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_progression_explanation_summarizes_q_margin_rewards_and_checkpoint() -> None:
    progression = generate_chords(
        [
            NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
            NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
        ],
        key="C major",
        emotion_vector=EmotionVector(valence=0.3, arousal=0.2),
        top_k=1,
    )[0]

    assert "average chosen Q=" in progression.explanation
    assert "weakest position" in progression.explanation
    assert "reward totals:" in progression.explanation
    assert "epoch14_reward4.298_mle_loss262.858_beta0.700.pth" in progression.explanation
