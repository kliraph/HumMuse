"""Service entrypoint for melody sketchpad inference."""

from __future__ import annotations

from ml.melody_sketchpad.pipeline import run_melody_pipeline
from shared.schemas import ChordSuggestion, ExplanationPart, MelodyNote


def generate_from_hum(
    audio_bytes: bytes, tempo_bpm: int | None
) -> tuple[list[MelodyNote], list[ChordSuggestion], list[ExplanationPart], bytes]:
    result = run_melody_pipeline(audio_bytes, tempo_bpm=tempo_bpm)
    return result.melody, result.chords, result.explanation, result.midi_bytes
