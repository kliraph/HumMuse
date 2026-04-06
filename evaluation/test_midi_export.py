"""Tests for MIDI export helpers."""

from __future__ import annotations

from ml.melody_sketchpad.midi_export import notes_to_midi
from shared.schemas import MelodyNote


def test_notes_to_midi_returns_standard_midi_bytes() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.0, duration_beats=1.0, velocity=96),
        MelodyNote(pitch="E4", start_beat=1.0, duration_beats=1.0, velocity=98),
        MelodyNote(pitch="G4", start_beat=2.0, duration_beats=1.0, velocity=100),
    ]

    midi_bytes = notes_to_midi(notes, tempo_bpm=120)

    assert midi_bytes.startswith(b"MThd")
    assert b"MTrk" in midi_bytes
    assert len(midi_bytes) > 32


def test_notes_to_midi_contains_note_pitches_from_input_melody() -> None:
    notes = [
        MelodyNote(pitch="C4", start_beat=0.0, duration_beats=0.5, velocity=90),
        MelodyNote(pitch="A4", start_beat=0.5, duration_beats=0.5, velocity=90),
    ]

    midi_bytes = notes_to_midi(notes, tempo_bpm=96)

    assert bytes([60]) in midi_bytes
    assert bytes([69]) in midi_bytes
