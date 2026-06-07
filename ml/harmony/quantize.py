"""Beat-grid quantization of a melody COPY for the chord model only.

The melody artifact the user keeps stays unquantized — see the note in
``ml/melody_sketchpad/pipeline.py`` explaining why snapping the *playable*
melody to a grid was dropped (it crushes hum rubato and creates phantom
polyphony at MIDI export).

This module is a separate, narrower use: it snaps a throwaway copy of the
melody onto a uniform beat grid purely so the DQN sees a clean
``bar_position`` / ``duration_type`` channel. PESTO onsets carry the natural
rubato of humming, which scatters the model's metric inputs across many
distinct bar positions and collapses chord output to a single sustained chord.
Snapping a copy to a beat grid restores the metric structure the model was
trained on. These notes are never exported, played back, or shown to the user —
they exist only as harmonizer input.

Inputs are assumed to already be in *beats* (as ``SessionState.melody_notes``
are after the melody pipeline's seconds->beats conversion), so no tempo is
required here. The grid is expressed in beats; the default is a quarter-note
(1.0-beat) grid, which is deliberately coarse — slow hums quantize more cleanly
at quarter resolution than at eighth resolution.
"""

from __future__ import annotations

from shared.schemas import NoteEvent

DEFAULT_GRID_BEATS = 1.0
DEFAULT_MIN_DURATION_BEATS = 1.0
_ROUND_DIGITS = 6


def notes_to_harmony_grid(
    note_events: list[NoteEvent],
    *,
    grid_beats: float = DEFAULT_GRID_BEATS,
    min_duration_beats: float = DEFAULT_MIN_DURATION_BEATS,
) -> list[NoteEvent]:
    """Snap a beat-valued melody copy onto a uniform grid for chord inference.

    ``onset`` and ``duration`` are assumed to be in beats. Returns a new list
    sorted by onset; the input events are not mutated. Durations below
    ``min_duration_beats`` are clamped up so every note keeps the positive
    duration the schema requires and contributes one chord position.
    """

    if grid_beats <= 0:
        raise ValueError("grid_beats must be positive")
    if min_duration_beats <= 0:
        raise ValueError("min_duration_beats must be positive")

    snapped: list[NoteEvent] = []
    for event in note_events:
        onset = _snap(float(event.onset), grid_beats)
        duration = _snap(float(event.duration), grid_beats)
        if duration < min_duration_beats:
            duration = round(min_duration_beats, _ROUND_DIGITS)
        snapped.append(event.model_copy(update={"onset": onset, "duration": duration}))
    snapped.sort(key=lambda event: event.onset)
    return snapped


def _snap(value: float, grid_beats: float) -> float:
    return round(round(max(0.0, value) / grid_beats) * grid_beats, _ROUND_DIGITS)
