"""Primer MIDI adapter for melody continuation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from midiutil import MIDIFile

from shared.schemas import NoteEvent

_TRACK = 0
_CHANNEL = 0
_TRACK_NAME = "HumMuse continuation primer"


def notes_to_prompt_midi(
    notes: list[NoteEvent],
    tempo: float,
    out_path: Path | str | None = None,
) -> Path:
    """
    Write NoteEvent primer notes to a monophonic MIDI file and return its path.

    NoteEvent timing is interpreted as seconds and converted to beats using
    `tempo`. The first note is normalized to beat 0 so the continuation model
    sees the melodic primer, not leading capture silence.
    """
    if tempo <= 0:
        raise ValueError(f"tempo must be positive, got {tempo}")

    if out_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="hummuse_primer_", suffix=".mid", delete=False)
        path = Path(handle.name)
        handle.close()
    else:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    midi = MIDIFile(1, deinterleave=False)
    midi.addTrackName(_TRACK, 0, _TRACK_NAME)
    midi.addTempo(_TRACK, 0, float(tempo))

    ordered = sorted(notes, key=lambda note: (note.onset, note.pitch, note.duration))
    first_onset = ordered[0].onset if ordered else 0.0
    beats_per_second = float(tempo) / 60.0

    for note in ordered:
        start_beats = max(0.0, (float(note.onset) - float(first_onset)) * beats_per_second)
        duration_beats = max(1e-3, float(note.duration) * beats_per_second)
        midi.addNote(
            track=_TRACK,
            channel=_CHANNEL,
            pitch=_clamp_int(note.pitch, 0, 127),
            time=start_beats,
            duration=duration_beats,
            volume=_clamp_int(note.velocity, 1, 127),
        )

    with path.open("wb") as file:
        midi.writeFile(file)
    return path


def _clamp_int(value: int | float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))
