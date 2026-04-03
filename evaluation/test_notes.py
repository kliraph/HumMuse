"""Tests for melody note conversion helpers."""

from __future__ import annotations

from ml.melody_sketchpad.notes import basic_pitch_notes_to_events, confidence_to_velocity


def test_basic_pitch_notes_to_events_maps_all_noteevent_fields_in_onset_order() -> None:
    note_events = [
        (0.60, 0.92, 71, 0.31, None),
        (0.15, 0.55, 69, 0.82, None),
    ]

    events = basic_pitch_notes_to_events(note_events)

    assert len(events) == 2
    assert events[0].pitch == 69
    assert events[0].onset == 0.15
    assert round(events[0].duration, 2) == 0.40
    assert events[0].confidence == 0.82
    assert events[0].velocity == 104
    assert events[1].pitch == 71
    assert events[1].onset == 0.60


def test_basic_pitch_notes_to_events_skips_nonpositive_durations_and_clamps_confidence() -> None:
    note_events = [
        (0.30, 0.30, 67, 0.5, None),
        (0.40, 0.20, 68, 0.9, None),
        (0.50, 0.75, 72, 1.5, None),
        (0.80, 1.00, 74, -0.5, None),
    ]

    events = basic_pitch_notes_to_events(note_events)

    assert len(events) == 2
    assert events[0].pitch == 72
    assert events[0].confidence == 1.0
    assert events[0].velocity == 127
    assert events[1].pitch == 74
    assert events[1].confidence == 0.0
    assert events[1].velocity == 1


def test_confidence_to_velocity_stays_within_midi_bounds() -> None:
    assert confidence_to_velocity(-1.0) == 1
    assert confidence_to_velocity(0.5) == 64
    assert confidence_to_velocity(2.0) == 127
