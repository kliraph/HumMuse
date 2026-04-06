"""Quantization helpers for sketch melody notes."""

from __future__ import annotations

from math import floor
from typing import Literal

from shared.schemas import MelodyNote

GridName = Literal["1/8", "1/16", "swing"]

_GRID_STEP = {
    "1/8": 0.5,
    "1/16": 0.25,
}
_SWING_OFFSET = 2.0 / 3.0
_ROUND_DIGITS = 6
_TOLERANCE = 1e-6


def quantize_notes(notes: list[MelodyNote], *, grid: GridName = "1/8") -> list[MelodyNote]:
    """Snap note starts and ends to a musical grid while preserving note order."""
    _validate_grid(grid)
    return [_quantize_note(note, grid=grid) for note in notes]


def _quantize_note(note: MelodyNote, *, grid: GridName) -> MelodyNote:
    start = _quantize_position(note.start_beat, grid=grid)
    end = _quantize_position(note.start_beat + note.duration_beats, grid=grid)
    if end <= start + _TOLERANCE:
        end = _next_grid_position(start, grid=grid)

    return note.model_copy(
        update={
            "start_beat": start,
            "duration_beats": round(end - start, _ROUND_DIGITS),
        }
    )


def _quantize_position(position: float, *, grid: GridName) -> float:
    point = max(0.0, float(position))
    if grid in _GRID_STEP:
        step = _GRID_STEP[grid]
        snapped = round(point / step) * step
        return round(snapped, _ROUND_DIGITS)

    beat = floor(point)
    candidates = {
        round(max(0.0, candidate), _ROUND_DIGITS)
        for beat_start in range(max(0, beat - 1), beat + 2)
        for candidate in (float(beat_start), beat_start + _SWING_OFFSET)
    }
    return min(candidates, key=lambda candidate: (abs(candidate - point), candidate))


def _next_grid_position(position: float, *, grid: GridName) -> float:
    point = round(max(0.0, float(position)), _ROUND_DIGITS)
    if grid in _GRID_STEP:
        return round(point + _GRID_STEP[grid], _ROUND_DIGITS)

    beat = floor(point)
    beat_start = round(float(beat), _ROUND_DIGITS)
    swing = round(beat + _SWING_OFFSET, _ROUND_DIGITS)
    next_beat = round(float(beat + 1), _ROUND_DIGITS)

    if abs(point - beat_start) <= _TOLERANCE:
        return swing
    if abs(point - swing) <= _TOLERANCE:
        return next_beat
    if point < swing:
        return swing
    return next_beat


def _validate_grid(grid: GridName) -> None:
    if grid not in {"1/8", "1/16", "swing"}:
        raise ValueError(f"Unsupported quantization grid: {grid}")
