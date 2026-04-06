"""Melody profile analysis helpers."""

from __future__ import annotations

from shared.schemas import MelodyProfile, NoteEvent


def build_melody_profile(notes: list[NoteEvent]) -> MelodyProfile:
    """Summarize interval motion, density, range, and contour for note events."""
    if not notes:
        return MelodyProfile(
            interval_histogram=[1.0, 0.0, 0.0, 0.0, 0.0],
            rhythmic_density=0.0,
            pitch_range=(0, 0),
            contour="rising",
        )

    ordered = sorted(notes, key=lambda note: note.onset)
    pitches = [note.pitch for note in ordered]
    intervals = [curr - prev for prev, curr in zip(pitches, pitches[1:])]
    buckets = [0, 0, 0, 0, 0]
    for interval in intervals:
        if interval <= -3:
            buckets[0] += 1
        elif interval < 0:
            buckets[1] += 1
        elif interval == 0:
            buckets[2] += 1
        elif interval < 3:
            buckets[3] += 1
        else:
            buckets[4] += 1

    total_intervals = sum(buckets) or 1
    total_beats = max(note.onset + note.duration for note in ordered)
    return MelodyProfile(
        interval_histogram=[bucket / total_intervals for bucket in buckets],
        rhythmic_density=len(ordered) / max(total_beats, 1.0),
        pitch_range=(min(pitches), max(pitches)),
        contour=classify_contour(pitches),
    )


def classify_contour(pitches: list[int]) -> str:
    """Classify the melodic contour from an ordered pitch sequence."""
    if len(pitches) < 3:
        return "rising" if pitches[-1] >= pitches[0] else "falling"

    peak = max(range(len(pitches)), key=pitches.__getitem__)
    valley = min(range(len(pitches)), key=pitches.__getitem__)
    if 0 < peak < len(pitches) - 1 and pitches[0] < pitches[peak] and pitches[-1] < pitches[peak]:
        return "arch"
    if 0 < valley < len(pitches) - 1 and pitches[0] > pitches[valley] and pitches[-1] > pitches[valley]:
        return "valley"
    return "rising" if pitches[-1] >= pitches[0] else "falling"
