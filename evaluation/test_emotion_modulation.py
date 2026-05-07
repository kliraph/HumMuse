"""Phase 4C-A.3 tests for emotion-conditioned DQN pitch-class Q heads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.harmony.dqn import DEFAULT_CHECKPOINT, generate_chords
from ml.harmony.emotion_modulation import emotion_bias_rationale, emotion_bias_vector, emotion_q_bias, lerdahl_tension
from shared.schemas import EmotionVector, NoteEvent


def test_high_valence_adds_major_triad_q_bias() -> None:
    q_pc = torch.zeros(13)
    biased = emotion_q_bias(q_pc, valence=1.0, arousal=0.0, key="C major")

    assert biased[4] > q_pc[4]
    assert biased[3] < q_pc[3]
    assert biased[0] == pytest.approx(q_pc[0])
    assert biased[7] == pytest.approx(q_pc[7])
    assert biased[12] == pytest.approx(q_pc[12])


def test_low_valence_adds_minor_triad_q_bias() -> None:
    q_pc = torch.zeros(13)
    biased = emotion_q_bias(q_pc, valence=-1.0, arousal=0.0, key="C major")

    assert biased[3] > q_pc[3]
    assert biased[4] < q_pc[4]
    assert biased[12] == pytest.approx(q_pc[12])


def test_low_valence_shifts_argmax_toward_minor_triad_pc() -> None:
    q_pc = torch.zeros(13)
    q_pc[4] = 0.4

    biased = emotion_q_bias(q_pc, valence=-1.0, arousal=0.0, key="C major")

    assert int(torch.argmax(biased).item()) in {0, 3, 7}
    assert int(torch.argmax(biased).item()) == 3


def test_high_arousal_scales_absolute_q_bias() -> None:
    q_pc = torch.zeros(13)
    neutral_arousal = emotion_bias_vector(q_pc, valence=0.5, arousal=0.0, key="C major")
    high_arousal = emotion_bias_vector(q_pc, valence=0.5, arousal=1.0, key="C major")

    assert high_arousal.abs().sum() > neutral_arousal.abs().sum()


def test_arousal_adds_independent_tension_bias_at_zero_valence() -> None:
    q_pc = torch.zeros(13)

    high_arousal = emotion_bias_vector(q_pc, valence=0.0, arousal=1.0, key="C major")
    no_arousal = emotion_bias_vector(q_pc, valence=0.0, arousal=0.0, key="C major")

    assert high_arousal.abs().sum() > 0.0
    assert no_arousal.abs().sum() == pytest.approx(0.0)
    assert high_arousal[1] > high_arousal[0]
    assert lerdahl_tension("C major")[1] > lerdahl_tension("C major")[0]


def test_zero_emotion_rationale_reports_no_bias() -> None:
    assert "No bias applied" in emotion_bias_rationale(valence=0.0, arousal=0.0, key="C major")
    assert "arousal tension" in emotion_bias_rationale(valence=0.0, arousal=1.0, key="C major")


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_generated_distribution_records_emotion_q_bias() -> None:
    melody = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
    ]

    progression = generate_chords(
        melody,
        key="C major",
        emotion_vector=EmotionVector(valence=0.7, arousal=0.8),
        top_k=1,
    )[0]
    bias = progression.native_distributions[0].emotion_bias

    assert bias is not None
    assert bias.applied is True
    assert len(bias.q_delta_per_pc) == 13
    assert "Q bias" in bias.rationale


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_zero_valence_with_arousal_records_tension_emotion_bias() -> None:
    progression = generate_chords(
        [NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95)],
        key="C major",
        emotion_vector=EmotionVector(valence=0.0, arousal=1.0),
        top_k=1,
    )[0]

    bias = progression.native_distributions[0].emotion_bias
    assert bias is not None
    assert bias.applied is True
    assert any(abs(delta) > 1e-12 for delta in bias.q_delta_per_pc)


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_zero_valence_zero_arousal_does_not_record_noop_emotion_bias() -> None:
    progression = generate_chords(
        [NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95)],
        key="C major",
        emotion_vector=EmotionVector(valence=0.0, arousal=0.0),
        top_k=1,
    )[0]

    assert progression.native_distributions[0].emotion_bias is None
