"""Service entrypoint for melody sketchpad mock inference."""

from __future__ import annotations

from ml.melody_sketchpad.chord_suggest import suggest_chords
from ml.melody_sketchpad.midi_render import render_midi_stub
from ml.melody_sketchpad.pitch import seed_from_size
from ml.melody_sketchpad.preprocess import audio_size
from ml.melody_sketchpad.quantize import quantize_notes
from shared.schemas import ChordSuggestion, ExplanationPart, MelodyNote


def generate_from_hum(
    audio_bytes: bytes, tempo_bpm: int | None
) -> tuple[list[MelodyNote], list[ChordSuggestion], list[ExplanationPart], bytes]:
    size = audio_size(audio_bytes)
    seed = seed_from_size(size)
    tempo = tempo_bpm or 100
    notes = [
        MelodyNote(pitch="C4", start_beat=0, duration_beats=1, velocity=90 + seed),
        MelodyNote(pitch="E4", start_beat=1, duration_beats=1, velocity=92 + seed),
        MelodyNote(pitch="G4", start_beat=2, duration_beats=1, velocity=95 + seed),
        MelodyNote(pitch="A4" if tempo > 110 else "F4", start_beat=3, duration_beats=1, velocity=97 + seed),
    ]
    melody = quantize_notes(notes)
    chords = suggest_chords()
    explanation = [
        ExplanationPart(
            title="Contour",
            detail="The phrase rises to a mild peak and resolves back for singable repetition.",
        ),
        ExplanationPart(
            title="Harmony fit",
            detail="Suggested chords stay diatonic to keep the progression stable for a first sketch.",
        ),
    ]
    midi_bytes = render_midi_stub(size)
    return melody, chords, explanation, midi_bytes
