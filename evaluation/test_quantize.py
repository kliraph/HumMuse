"""Tests for melody quantization helpers."""

from __future__ import annotations

import pytest

from ml.melody_sketchpad.quantize import quantize_notes
from shared.schemas import MelodyNote


def test_quantize_notes_snaps_to_eighth_note_grid() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.24, duration_beats=0.61, velocity=100),
        MelodyNote(pitch="E4", start_beat=1.74, duration_beats=0.52, velocity=96),
    ]

    quantized = quantize_notes(notes, grid="1/8")

    assert quantized[0].start_beat == 0.0
    assert quantized[0].duration_beats == 1.0
    assert quantized[1].start_beat == 1.5
    assert quantized[1].duration_beats == 1.0


def test_quantize_notes_snaps_to_sixteenth_grid() -> None:
    notes = [
        MelodyNote(pitch="A4", start_beat=0.37, duration_beats=0.41, velocity=92),
        MelodyNote(pitch="B4", start_beat=1.18, duration_beats=0.33, velocity=88),
    ]

    quantized = quantize_notes(notes, grid="1/16")

    assert quantized[0].start_beat == 0.25
    assert quantized[0].duration_beats == 0.5
    assert quantized[1].start_beat == 1.25
    assert quantized[1].duration_beats == 0.25


def test_quantize_notes_snaps_offbeats_to_swing_grid() -> None:
    notes = [
        MelodyNote(pitch="G4", start_beat=0.46, duration_beats=0.41, velocity=90),
        MelodyNote(pitch="A4", start_beat=1.61, duration_beats=0.38, velocity=90),
    ]

    quantized = quantize_notes(notes, grid="swing")

    assert quantized[0].start_beat == pytest.approx(2.0 / 3.0, rel=0, abs=1e-6)
    assert quantized[0].duration_beats == pytest.approx(1.0 / 3.0, rel=0, abs=1e-6)
    assert quantized[1].start_beat == pytest.approx(1.0 + (2.0 / 3.0), rel=0, abs=1e-6)
    assert quantized[1].duration_beats == pytest.approx(1.0 / 3.0, rel=0, abs=1e-6)


def test_quantize_notes_preserves_minimum_duration_when_rounding_collapses() -> None:
    notes = [MelodyNote(pitch="D4", start_beat=0.12, duration_beats=0.08, velocity=100)]

    quantized = quantize_notes(notes, grid="1/16")

    assert quantized[0].start_beat == 0.0
    assert quantized[0].duration_beats == 0.25


def test_quantize_notes_rejects_unknown_grid() -> None:
    notes = [MelodyNote(pitch="C4", start_beat=0.0, duration_beats=1.0, velocity=100)]

    with pytest.raises(ValueError, match="Unsupported quantization grid"):
        quantize_notes(notes, grid="triplet")  # type: ignore[arg-type]
