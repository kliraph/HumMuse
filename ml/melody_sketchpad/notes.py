"""Helpers for converting extracted melody notes into shared note events."""

from __future__ import annotations

from shared.schemas import MelodyNote, NoteEvent

_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

_PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MIN_DURATION_BEATS = 1e-3

def confidence_to_velocity(confidence: float) -> int:
    clipped = max(0.0, min(1.0, float(confidence)))
    return max(1, min(127, int(round(clipped * 126)) + 1))


def melody_notes_to_events(notes: list[MelodyNote]) -> list[NoteEvent]:
    return [
        NoteEvent(
            pitch=pitch_to_midi(note.pitch),
            onset=note.start_beat,
            duration=note.duration_beats,
            velocity=note.velocity,
            confidence=0.92,
        )
        for note in notes
    ]


def pitch_to_midi(pitch: str) -> int:
    note = pitch[:-1]
    octave = int(pitch[-1])
    return (octave + 1) * 12 + _PITCH_CLASS[note]


def midi_to_pitch_string(midi_pitch: int) -> str:
    midi = max(0, min(127, int(midi_pitch)))
    octave = (midi // 12) - 1
    return f"{_PITCH_CLASS_LABELS[midi % 12]}{octave}"


def note_events_to_melody_notes(
    note_events: list[NoteEvent], *, tempo_bpm: int | None
) -> list[MelodyNote]:
    """Convert second-based NoteEvents into beat-based MelodyNotes for the pipeline."""
    tempo = float(tempo_bpm) if tempo_bpm else 100.0
    beats_per_second = tempo / 60.0
    melody: list[MelodyNote] = []
    for event in note_events:
        start_beat = max(0.0, float(event.onset) * beats_per_second)
        duration_beats = max(_MIN_DURATION_BEATS, float(event.duration) * beats_per_second)
        melody.append(
            MelodyNote(
                pitch=midi_to_pitch_string(int(event.pitch)),
                start_beat=start_beat,
                duration_beats=duration_beats,
                velocity=int(event.velocity),
            )
        )
    return melody
