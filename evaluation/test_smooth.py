"""Tests for melody smoothing helpers."""

from __future__ import annotations

from ml.melody_sketchpad.smooth import smooth_notes
from shared.schemas import MelodyNote


def test_smooth_notes_merges_touching_notes_within_one_semitone() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.0, duration_beats=0.5, velocity=92),
        MelodyNote(pitch="C#4", start_beat=0.5, duration_beats=0.5, velocity=101),
        MelodyNote(pitch="E4", start_beat=1.5, duration_beats=0.5, velocity=96),
    ]

    smoothed = smooth_notes(notes)

    assert len(smoothed) == 2
    assert smoothed[0].pitch == "C4"
    assert smoothed[0].start_beat == 0.0
    assert smoothed[0].duration_beats == 1.0
    assert smoothed[0].velocity == 101
    assert smoothed[1].pitch == "E4"


def test_smooth_notes_does_not_merge_when_gap_exists() -> None:
    notes = [
        MelodyNote(pitch="G4", start_beat=0.0, duration_beats=0.5, velocity=90),
        MelodyNote(pitch="G#4", start_beat=0.75, duration_beats=0.5, velocity=90),
    ]

    smoothed = smooth_notes(notes)

    assert len(smoothed) == 2


def test_smooth_notes_removes_short_low_velocity_ghosts() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.0, duration_beats=1.0, velocity=90),
        MelodyNote(pitch="D4", start_beat=1.0, duration_beats=0.25, velocity=18),
        MelodyNote(pitch="E4", start_beat=1.25, duration_beats=0.5, velocity=90),
    ]

    smoothed = smooth_notes(notes)

    assert len(smoothed) == 2
    assert [note.pitch for note in smoothed] == ["C4", "E4"]


def test_smooth_notes_keeps_short_accented_notes() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.0, duration_beats=1.0, velocity=90),
        MelodyNote(pitch="D4", start_beat=1.0, duration_beats=0.25, velocity=72),
    ]

    smoothed = smooth_notes(notes)

    assert len(smoothed) == 2
    assert smoothed[1].pitch == "D4"
