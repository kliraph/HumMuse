"""Key detection using the Krumhansl-Schmuckler algorithm."""

from __future__ import annotations

import math
from typing import Final

from shared.schemas import NoteEvent

_NOTE_NAMES: Final[tuple[str, ...]] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_PROFILE: Final[tuple[float, ...]] = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE: Final[tuple[float, ...]] = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def detect_key(note_events: list[NoteEvent]) -> str | None:
    """Return the best-fitting tonal key for the given note events."""
    if not note_events:
        return None

    pitch_class_weights = _pitch_class_histogram(note_events)
    if sum(1 for weight in pitch_class_weights if weight > 0.0) < 2:
        return None

    tonic_scores: list[tuple[float, int, str]] = []
    first_pc = int(note_events[0].pitch) % 12
    last_pc = int(max(note_events, key=lambda note: note.onset + note.duration).pitch) % 12

    for tonic in range(12):
        major_score = _pearson_correlation(pitch_class_weights, _rotate_profile(_MAJOR_PROFILE, tonic))
        minor_score = _pearson_correlation(pitch_class_weights, _rotate_profile(_MINOR_PROFILE, tonic))
        tonic_scores.append((_apply_tonic_bonus(major_score, tonic, first_pc, last_pc), tonic, "major"))
        tonic_scores.append((_apply_tonic_bonus(minor_score, tonic, first_pc, last_pc), tonic, "minor"))

    _, tonic, mode = max(tonic_scores, key=lambda item: item[0])
    return f"{_NOTE_NAMES[tonic]} {mode}"


def _pitch_class_histogram(note_events: list[NoteEvent]) -> list[float]:
    histogram = [0.0] * 12
    for note in note_events:
        weight = max(float(note.duration), 0.0) * max(float(note.confidence), 0.1)
        histogram[int(note.pitch) % 12] += weight
    return histogram


def _rotate_profile(profile: tuple[float, ...], tonic: int) -> list[float]:
    tonic = tonic % 12
    return [float(profile[(index - tonic) % 12]) for index in range(12)]


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((lhs - left_mean) * (rhs - right_mean) for lhs, rhs in zip(left, right))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return float("-inf")
    return numerator / (left_norm * right_norm)


def _apply_tonic_bonus(score: float, tonic: int, first_pc: int, last_pc: int) -> float:
    bonus = 0.0
    if tonic == first_pc:
        bonus += 0.05
    if tonic == last_pc:
        bonus += 0.12
    return score + bonus
