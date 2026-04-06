"""MIDI export helpers for sketch melody output."""

from __future__ import annotations

from io import BytesIO
from typing import Final

from midiutil import MIDIFile

from ml.melody_sketchpad.notes import pitch_to_midi
from shared.schemas import MelodyNote

_DEFAULT_TEMPO_BPM: Final[int] = 100
_DEFAULT_TRACK_NAME: Final[str] = "HumMuse melody"
_DEFAULT_CHANNEL: Final[int] = 0
_DEFAULT_TRACK: Final[int] = 0
_DEFAULT_VOLUME: Final[int] = 100


def notes_to_midi(notes: list[MelodyNote], *, tempo_bpm: int | None = None) -> bytes:
    """Render melody notes into a standard single-track MIDI file."""
    midi = MIDIFile(1, deinterleave=False)
    tempo = int(tempo_bpm or _DEFAULT_TEMPO_BPM)
    midi.addTrackName(_DEFAULT_TRACK, 0, _DEFAULT_TRACK_NAME)
    midi.addTempo(_DEFAULT_TRACK, 0, tempo)

    for note in sorted(notes, key=lambda item: (item.start_beat, item.duration_beats, item.pitch)):
        midi.addNote(
            track=_DEFAULT_TRACK,
            channel=_DEFAULT_CHANNEL,
            pitch=pitch_to_midi(note.pitch),
            time=float(note.start_beat),
            duration=float(note.duration_beats),
            volume=int(note.velocity),
        )

    buffer = BytesIO()
    midi.writeFile(buffer)
    return buffer.getvalue()
