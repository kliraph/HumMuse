"""Melody smoothing helpers for post-quantized note sequences."""

from __future__ import annotations

from typing import Final

from ml.melody_sketchpad.notes import pitch_to_midi
from shared.schemas import MelodyNote

_ROUND_DIGITS: Final[int] = 6
_TIME_TOLERANCE: Final[float] = 1e-6
_DEFAULT_GHOST_DURATION_BEATS: Final[float] = 0.25
_DEFAULT_GHOST_VELOCITY: Final[int] = 24
_DEFAULT_MAX_MERGE_INTERVAL: Final[int] = 1


def smooth_notes(
    notes: list[MelodyNote],
    *,
    ghost_duration_beats: float = _DEFAULT_GHOST_DURATION_BEATS,
    ghost_velocity: int = _DEFAULT_GHOST_VELOCITY,
    max_merge_interval_semitones: int = _DEFAULT_MAX_MERGE_INTERVAL,
) -> list[MelodyNote]:
    """Remove likely extraction ghosts and merge near-unison adjacent notes."""
    filtered = [
        note
        for note in sorted(notes, key=lambda item: (item.start_beat, item.duration_beats))
        if not _is_ghost(
            note,
            ghost_duration_beats=ghost_duration_beats,
            ghost_velocity=ghost_velocity,
        )
    ]
    if not filtered:
        return []

    smoothed: list[MelodyNote] = [filtered[0]]
    for note in filtered[1:]:
        current = smoothed[-1]
        if _should_merge(current, note, max_merge_interval_semitones=max_merge_interval_semitones):
            smoothed[-1] = _merge_notes(current, note)
            continue
        smoothed.append(note)

    return smoothed


def _is_ghost(note: MelodyNote, *, ghost_duration_beats: float, ghost_velocity: int) -> bool:
    return note.duration_beats <= ghost_duration_beats and note.velocity <= ghost_velocity


def _should_merge(
    first: MelodyNote,
    second: MelodyNote,
    *,
    max_merge_interval_semitones: int,
) -> bool:
    first_end = first.start_beat + first.duration_beats
    if second.start_beat > first_end + _TIME_TOLERANCE:
        return False
    return abs(pitch_to_midi(first.pitch) - pitch_to_midi(second.pitch)) <= max_merge_interval_semitones


def _merge_notes(first: MelodyNote, second: MelodyNote) -> MelodyNote:
    merged_end = max(
        first.start_beat + first.duration_beats,
        second.start_beat + second.duration_beats,
    )
    return first.model_copy(
        update={
            "duration_beats": round(merged_end - first.start_beat, _ROUND_DIGITS),
            "velocity": max(first.velocity, second.velocity),
        }
    )
