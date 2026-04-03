"""Helpers for converting extracted melody notes into shared note events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shared.schemas import NoteEvent

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
