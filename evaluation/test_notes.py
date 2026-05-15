"""Tests for melody note conversion helpers."""

from __future__ import annotations

from ml.melody_sketchpad.notes import (
    confidence_to_velocity,
    midi_to_pitch_string,
    note_events_to_melody_notes,
)
from shared.schemas import NoteEvent


def test_confidence_to_velocity_stays_within_midi_bounds() -> None:
    assert confidence_to_velocity(-1.0) == 1
    assert confidence_to_velocity(0.5) == 64
    assert confidence_to_velocity(2.0) == 127


def test_midi_to_pitch_string_for_common_pitches() -> None:
    assert midi_to_pitch_string(60) == "C4"
    assert midi_to_pitch_string(69) == "A4"
    assert midi_to_pitch_string(61) == "C#4"
    assert midi_to_pitch_string(72) == "C5"
    assert midi_to_pitch_string(0) == "C-1"


def test_note_events_to_melody_notes_converts_seconds_to_beats_at_given_tempo() -> None:
    events = [
        NoteEvent(pitch=60, onset=0.0, duration=0.6, velocity=100, confidence=0.9),
        NoteEvent(pitch=64, onset=0.6, duration=0.3, velocity=95, confidence=0.85),
    ]
    melody = note_events_to_melody_notes(events, tempo_bpm=100)

    assert len(melody) == 2
    assert melody[0].pitch == "C4"
    assert melody[0].start_beat == 0.0
    assert round(melody[0].duration_beats, 2) == 1.0  # 0.6s × (100/60)
    assert melody[1].pitch == "E4"
    assert round(melody[1].start_beat, 2) == 1.0
    assert round(melody[1].duration_beats, 2) == 0.5


def test_note_events_to_melody_notes_floors_duration_to_minimum() -> None:
    events = [NoteEvent(pitch=60, onset=0.0, duration=1e-9, velocity=80, confidence=0.5)]
    melody = note_events_to_melody_notes(events, tempo_bpm=100)
    assert melody[0].duration_beats > 0
