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
# Same-pitch gap-bridging was retired (2026-05-20) when onset-aware
# segmentation landed in pitch._compute_onset_frames / _segment_pitch_frames.
# The two features address the same problem (breath/glottal dips inside what
# the singer intended as one held note) but onset detection is much sharper:
# it preserves the genuine repeated-pitch notes ("C4 C4") that the old
# bridging code merged away. Default is 0 so the same-pitch branch is a
# no-op; keep the parameter to preserve the API for any external callers.
_DEFAULT_SAME_PITCH_MERGE_GAP_BEATS: Final[float] = 0.0


def smooth_notes(
    notes: list[MelodyNote],
    *,
    ghost_duration_beats: float = _DEFAULT_GHOST_DURATION_BEATS,
    ghost_velocity: int = _DEFAULT_GHOST_VELOCITY,
    max_merge_interval_semitones: int = _DEFAULT_MAX_MERGE_INTERVAL,
    same_pitch_merge_gap_beats: float = _DEFAULT_SAME_PITCH_MERGE_GAP_BEATS,
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
        if _should_merge(
            current,
            note,
            max_merge_interval_semitones=max_merge_interval_semitones,
            same_pitch_merge_gap_beats=same_pitch_merge_gap_beats,
        ):
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
    same_pitch_merge_gap_beats: float,
) -> bool:
    """Decide whether two adjacent notes should be merged into one.

    Same-pitch merging is disabled: onset-aware segmentation already preserves
    genuine repeated notes (e.g. "E4 E4"), and the old same-pitch bridge would
    erase exactly those. The remaining merge path only collapses *near-unison
    pitch wobble* across the 1-semitone segmenter jump-threshold — abutting
    notes 1 semitone apart that PESTO emitted as a brief artefact of vibrato.
    The ``same_pitch_merge_gap_beats`` argument is kept for API compatibility
    but is no longer consulted.
    """
    interval = abs(pitch_to_midi(first.pitch) - pitch_to_midi(second.pitch))
    if interval == 0:
        return False  # let onset segmentation own repeated-note boundaries
    if interval > max_merge_interval_semitones:
        return False
    first_end = first.start_beat + first.duration_beats
    gap = second.start_beat - first_end
    return gap <= _TIME_TOLERANCE  # only merge abutting near-unison wobble


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
