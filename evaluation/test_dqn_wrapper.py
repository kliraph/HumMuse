"""Tests for the DQN harmony wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ml.harmony.dqn as dqn
from ml.harmony.dqn import (
    DEFAULT_CHECKPOINT,
    build_dqn_state,
    decode,
    encode_chord_to_20dim,
    generate_chords,
    load_model,
    notes_to_events,
    start_chord_one_hot,
)
from shared.schemas import EmotionVector, NoteEvent


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_generate_chords_returns_ranked_progressions_with_native_distributions() -> None:
    melody = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=72, onset=3.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=71, onset=4.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=67, onset=5.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=6.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=60, onset=7.0, duration=1.0, velocity=96, confidence=0.95),
    ]

    progressions = generate_chords(
        melody,
        key="C major",
        emotion_vector=EmotionVector(valence=0.7, arousal=0.4),
        top_k=3,
    )

    assert len(progressions) == 3
    assert all(progression.chords for progression in progressions)
    assert all(progression.native_distributions for progression in progressions)
    first_distribution = progressions[0].native_distributions[0]
    assert first_distribution.position == 0
    assert first_distribution.pitch_class_labels[-1] == "triad_sentinel"
    assert len(first_distribution.pitch_class_membership) == 13
    assert progressions[0].model_dump()["native_distributions"]


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_dqn_state_shape_and_disabled_noise_are_deterministic() -> None:
    melody = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
    ]
    events = notes_to_events(melody)
    state = build_dqn_state(events, start_chord_one_hot(), position=0)
    cached = load_model()

    first = cached.model.forward_with_dueling(state)
    second = cached.model.forward_with_dueling(state)

    assert tuple(state.shape) == (1, 1, 177)
    assert first[0].tolist() == second[0].tolist()
    assert first[5]["advantage_pc"].shape[-1] == 13


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_fixed_melody_q_values_are_stable_with_noise_disabled() -> None:
    melody = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=96, confidence=0.95),
    ]

    first = generate_chords(melody, key="C major", emotion_vector=None, top_k=1)[0]
    second = generate_chords(melody, key="C major", emotion_vector=None, top_k=1)[0]

    first_q = [distribution.q_values.model_dump() for distribution in first.native_distributions]
    second_q = [distribution.q_values.model_dump() for distribution in second.native_distributions]
    assert first_q == second_q


def test_sequential_generation_feeds_previous_chord_one_hot() -> None:
    events = [(60, 2, 0), (64, 2, 24)]

    class FixedModel:
        def __init__(self) -> None:
            self.states = []

        def forward_with_dueling(self, state, hidden=None):
            self.states.append(state.detach().clone())
            q1 = torch.tensor([[[-4.0]]])
            q2 = torch.tensor([[[0.0, 2.0]]])
            q4 = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]])
            q13 = torch.tensor([[[5.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0]]])
            dueling = {
                "value_rest": torch.tensor([[[-4.0]]]),
                "value_octave": torch.tensor([[[1.0]]]),
                "value_inversion": torch.tensor([[[1.0]]]),
                "value_pc": torch.tensor([[[1.0]]]),
                "advantage_rest": torch.tensor([[[0.0]]]),
                "advantage_octave": torch.tensor([[[-1.0, 1.0]]]),
                "advantage_inversion": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                "advantage_pc": q13,
            }
            return q1, q2, q4, q13, hidden, dueling

    model = FixedModel()
    rollout = dqn._rollout(model, events, key="C major", emotion_vector=None, candidate_rank=0)
    expected_previous = encode_chord_to_20dim(rollout["distributions"][0].selected)

    assert len(model.states) == 2
    assert model.states[1].view(-1)[-20:].tolist() == expected_previous


def test_decode_ranks_runner_ups_by_joint_template_q_not_pitch_window_offset() -> None:
    policy = {"rest": {"rest": 0.01}}
    q1 = torch.tensor([[[-4.0]]])
    q2 = torch.tensor([[[2.0, 0.0]]])
    q4 = torch.tensor([[[3.0, 2.5, 1.0, 0.0]]])
    q13 = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 11.0]]])

    best = decode(policy, q1, q2, q4, q13, candidate_rank=0)
    runner_up = decode(policy, q1, q2, q4, q13, candidate_rank=1)

    assert best["pcs"] == [0, 4, 7]
    assert runner_up["pcs"] == [0, 4, 7]
    assert runner_up["inversion"] == 1


