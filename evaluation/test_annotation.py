"""Phase 4C-A.4 tests for deterministic chord annotations."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.harmony.annotation import annotate_chord_positions
from ml.harmony.dqn import DEFAULT_CHECKPOINT, generate_chords
from ml.harmony.chord_symbols import derive_chord_symbol
from shared.schemas import ChordDistribution, EmotionVector, NoteEvent

PC_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel"]


def _distribution(pcs: list[int], inversion: int = 0, position: int = 0) -> ChordDistribution:
    membership = [0.001] * 13
    for pc in pcs:
        membership[pc] = 0.9
    membership[12] = 0.95 if len(pcs) == 3 else 0.001
    total = sum(membership)
    membership = [value / total for value in membership]
    return ChordDistribution(
        position=position,
        rest={"rest": 0.01, "chord": 0.99},
        octave={"2": 0.1, "3": 0.9},
        inversion={str(index): 1.0 if index == inversion else 0.0 for index in range(4)},
        pitch_class={label: membership[index] for index, label in enumerate(PC_LABELS[:12])},
        pitch_class_membership=membership,
        pitch_class_labels=PC_LABELS,
        selected={
            "is_rest": False,
            "octave": 3,
            "inversion": inversion,
            "pcs": sorted(pcs),
            "pc_names": [PC_LABELS[pc] for pc in sorted(pcs)],
        },
    )


def test_annotation_adds_symbol_function_roman_alignment_and_template_phrase() -> None:
    distributions = [
        _distribution([0, 4, 7], position=0),
        _distribution([0, 4, 7], position=1),
        _distribution([0, 4, 7], position=2),
    ]
    melody_events = [
        (60, 2, 0),
        (62, 2, 24),
        (64, 2, 12),
    ]
    derivations = [derive_chord_symbol(distribution, key="C major") for distribution in distributions]

    annotations = annotate_chord_positions(
        distributions,
        melody_events,
        key="C major",
        derivations=derivations,
    )

    assert len(annotations) == 3
    assert annotations[0].symbol == "C"
    assert annotations[0].function_label == "tonic"
    assert annotations[0].roman_numeral == "I"
    assert annotations[0].alignment_percentage == 1.0
    assert annotations[0].q_margin is not None
    assert annotations[0].value_score is not None
    assert annotations[0].strong_beat_notes[0]["is_chord_tone"] is True
    assert annotations[0].template_phrase == "tonic - stable home base"

    assert annotations[1].alignment_percentage == 0.0
    assert annotations[1].strong_beat_notes[0]["pitch_class_name"] == "D"
    assert annotations[1].strong_beat_notes[0]["is_chord_tone"] is False

    assert annotations[2].alignment_percentage == 0.0
    assert annotations[2].strong_beat_notes == []


def test_annotations_are_deterministic_and_round_trip_serializable() -> None:
    distribution = _distribution([9, 0, 4], inversion=1)
    melody_events = [(69, 2, 0)]
    first = annotate_chord_positions([distribution], melody_events, key="C major")
    second = annotate_chord_positions([distribution], melody_events, key="C major")

    assert first == second
    encoded = first[0].model_dump_json()
    assert "submediant - tonic substitute" in encoded


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT).exists(), reason="RL-Chord checkpoint is not available")
def test_dqn_wrapper_populates_annotations_for_each_generated_chord() -> None:
    melody = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=72, onset=3.0, duration=1.0, velocity=96, confidence=0.95),
    ]

    progression = generate_chords(
        melody,
        key="C major",
        emotion_vector=EmotionVector(valence=0.3, arousal=0.2),
        top_k=1,
    )[0]

    assert len(progression.chord_annotations) == len(progression.chords)
    annotation = progression.chord_annotations[0]
    assert annotation.symbol
    assert annotation.function_label
    assert annotation.roman_numeral
    assert annotation.q_margin is not None
    assert annotation.value_score is not None
    assert annotation.q_margin >= 0.0
    assert 0.0 <= annotation.alignment_percentage <= 1.0
    assert annotation.template_phrase


