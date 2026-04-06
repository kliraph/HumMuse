"""Helpers for converting extracted melody notes into shared note events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shared.schemas import MelodyNote, NoteEvent

_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

BasicPitchNoteTuple = tuple[float, float, int, float, Any]


def confidence_to_velocity(confidence: float) -> int:
    clipped = max(0.0, min(1.0, float(confidence)))
    return max(1, min(127, int(round(clipped * 126)) + 1))


def basic_pitch_notes_to_events(note_events: Iterable[BasicPitchNoteTuple]) -> list[NoteEvent]:
    """Convert Basic Pitch note tuples into ordered NoteEvent models."""
    extracted: list[NoteEvent] = []
    for start_time, end_time, pitch, amplitude, _ in sorted(note_events, key=lambda event: event[0]):
        duration = max(0.0, float(end_time) - float(start_time))
        if duration <= 0:
            continue

        confidence = max(0.0, min(1.0, float(amplitude)))
        extracted.append(
            NoteEvent(
                pitch=int(pitch),
                onset=float(start_time),
                duration=duration,
                velocity=confidence_to_velocity(confidence),
                confidence=confidence,
            )
        )

    return extracted


def melody_notes_to_events(notes: list[MelodyNote]) -> list[NoteEvent]:
    return [
        NoteEvent(
            pitch=pitch_to_midi(note.pitch),
            onset=note.start_beat,
            duration=note.duration_beats,
            velocity=note.velocity,
            confidence=0.92,
        )
        for note in notes
    ]


def pitch_to_midi(pitch: str) -> int:
    note = pitch[:-1]
    octave = int(pitch[-1])
    return (octave + 1) * 12 + _PITCH_CLASS[note]
