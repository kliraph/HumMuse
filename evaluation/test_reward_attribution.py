"""Tests for DQN reward attribution components."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.harmony.dqn import DEFAULT_CHECKPOINT, generate_chords
from ml.harmony.reward_attribution import (
    chord_tone_inclusion,
    compute_attribution,
    harmony_rule,
    lerdahl_basic_space_distance,
    progression_penalty,
)
from shared.schemas import NoteEvent


def _c_major() -> dict:
    return {
        "is_rest": False,
        "octave": 3,
        "inversion": 0,
        "pcs": [0, 4, 7],
        "pc_names": ["C", "E", "G"],
    }


def test_harmony_rule_rewards_consonant_melody_and_chord_tones() -> None:
    consonant = harmony_rule(_c_major(), (60, 2, 0), None, "C major")
    dissonant = harmony_rule(_c_major(), (61, 2, 0), None, "C major")

    assert consonant > dissonant


def test_progression_penalty_uses_continuous_basic_space_distance() -> None:
    repeated_downbeat = progression_penalty(_c_major(), (60, 2, 0), _c_major(), "C major")
    repeated_offbeat = progression_penalty(_c_major(), (60, 2, 24), _c_major(), "C major")
    distant_chromatic = progression_penalty(
        {"is_rest": False, "octave": 3, "inversion": 0, "pcs": [1, 3, 6], "pc_names": ["C#", "D#", "F#"]},
        (60, 2, 24),
        _c_major(),
        "C major",
    )
    dominant = progression_penalty(
        {"is_rest": False, "octave": 3, "inversion": 0, "pcs": [7, 11, 2], "pc_names": ["G", "B", "D"]},
        (60, 2, 24),
        _c_major(),
        "C major",
    )

    assert repeated_downbeat == -1.0
    assert repeated_offbeat == 1.0
    assert dominant == pytest.approx(-0.666667)
    assert distant_chromatic == pytest.approx(-0.833333)
    assert dominant > distant_chromatic
    assert lerdahl_basic_space_distance({0, 4, 7}, {7, 11, 2}, "C major") == 8


def test_chord_tone_inclusion_and_compute_attribution_shape() -> None:
    attribution = compute_attribution(_c_major(), (64, 2, 0), None, "C major")

    assert chord_tone_inclusion(_c_major(), (64, 2, 0), None, "C major") == 1.0
    assert set(attribution) == {
        "harmony_rule",
        "progression_penalty",
        "chord_tone_inclusion",
        "mutual_info",
        "key_fit_proxy",
    }
    assert attribution["mutual_info"] == 0.0
    assert attribution["key_fit_proxy"] > 0.0


def test_compute_attribution_matches_known_reference_value() -> None:
    attribution = compute_attribution(_c_major(), (60, 2, 0), None, "C major")

    assert attribution["harmony_rule"] == pytest.approx(14.333333)
    assert attribution["progression_penalty"] == 0.0
    assert attribution["chord_tone_inclusion"] == 1.0
    assert attribution["mutual_info"] == 0.0
    assert attribution["key_fit_proxy"] == 1.0


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_generated_distribution_records_reward_attribution() -> None:
    progression = generate_chords(
        [
            NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
            NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
        ],
        key="C major",
        emotion_vector=None,
        top_k=1,
    )[0]

    attribution = progression.native_distributions[0].reward_attribution
    assert attribution is not None
    assert isinstance(attribution.harmony_rule, float)
    assert attribution.mutual_info == 0.0
