"""Melody smoothing: drop low-confidence extraction ghosts."""

from __future__ import annotations

from typing import Final

from shared.schemas import MelodyNote

_DEFAULT_GHOST_DURATION_BEATS: Final[float] = 0.25
_DEFAULT_GHOST_VELOCITY: Final[int] = 24


def smooth_notes(
    notes: list[MelodyNote],
    *,
    ghost_duration_beats: float = _DEFAULT_GHOST_DURATION_BEATS,
    ghost_velocity: int = _DEFAULT_GHOST_VELOCITY,
) -> list[MelodyNote]:
    """Drop likely extraction ghosts — notes that are both short *and* quiet.

    Note merging used to live here too, but repeated-pitch and near-unison
    boundaries are now owned by the onset-aware segmenter
    (``pitch._segment_pitch_frames`` + voicing hysteresis), which has the
    frame-level context that smoothing lacked. A semitone is a real melodic
    step, so nothing here collapses distinct pitches. Order is preserved.
    """
    return [
        note
        for note in sorted(notes, key=lambda item: (item.start_beat, item.duration_beats))
        if not _is_ghost(
            note,
            ghost_duration_beats=ghost_duration_beats,
            ghost_velocity=ghost_velocity,
        )
    ]


def _is_ghost(note: MelodyNote, *, ghost_duration_beats: float, ghost_velocity: int) -> bool:
    return note.duration_beats <= ghost_duration_beats and note.velocity <= ghost_velocity
