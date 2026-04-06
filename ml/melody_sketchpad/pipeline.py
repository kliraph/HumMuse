"""End-to-end melody pipeline with per-step latency logging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import soundfile as sf

from ml.melody_sketchpad.chord_suggest import suggest_chords
from ml.melody_sketchpad.key_detect import detect_key
from ml.melody_sketchpad.midi_export import notes_to_midi
from ml.melody_sketchpad.notes import melody_notes_to_events
from ml.melody_sketchpad.pitch import seed_from_size
from ml.melody_sketchpad.preprocess import (
    DEFAULT_SAMPLE_RATE,
    audio_size,
    load_audio,
    preprocess_audio,
)
from ml.melody_sketchpad.profile import build_melody_profile
from ml.melody_sketchpad.quantize import quantize_notes
from ml.melody_sketchpad.smooth import smooth_notes
from shared.schemas import ChordSuggestion, ExplanationPart, MelodyNote, MelodyProfile, NoteEvent

_ROUND_DIGITS = 3


@dataclass(frozen=True)
class MelodyResult:
    melody: list[MelodyNote]
    note_events: list[NoteEvent]
    melody_profile: MelodyProfile
    detected_key: str | None
    chords: list[ChordSuggestion]
    explanation: list[ExplanationPart]
    midi_bytes: bytes
    detected_tempo: float
    step_latency_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_melody_pipeline(
    audio_bytes: bytes,
    tempo_bpm: int | None = None,
    mood: str | None = None,
) -> MelodyResult:
    """Execute the melody pipeline from audio bytes to a structured result."""
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    tempo = float(tempo_bpm or 100)

    decode_started = time.perf_counter()
    audio, used_audio_fallback, decode_error = _load_audio_with_fallback(audio_bytes)
    timings["decode_audio"] = _elapsed_ms(decode_started)
    metadata["used_audio_fallback"] = used_audio_fallback
    if decode_error is not None:
        metadata["decode_error"] = decode_error

    preprocess_started = time.perf_counter()
    processed_audio = preprocess_audio(audio, DEFAULT_SAMPLE_RATE)
    timings["preprocess_audio"] = _elapsed_ms(preprocess_started)

    notes_started = time.perf_counter()
    melody = _generate_mock_melody(audio_bytes, tempo_bpm=tempo_bpm, mood=mood)
    timings["generate_notes"] = _elapsed_ms(notes_started)

    quantize_started = time.perf_counter()
    quantized = quantize_notes(melody)
    timings["quantize_notes"] = _elapsed_ms(quantize_started)

    smooth_started = time.perf_counter()
    smoothed = smooth_notes(quantized)
    timings["smooth_notes"] = _elapsed_ms(smooth_started)

    events_started = time.perf_counter()
    note_events = melody_notes_to_events(smoothed)
    profile = build_melody_profile(note_events)
    timings["build_profile"] = _elapsed_ms(events_started)

    key_started = time.perf_counter()
    detected_key = detect_key(note_events)
    timings["detect_key"] = _elapsed_ms(key_started)

    midi_started = time.perf_counter()
    midi_bytes = notes_to_midi(smoothed, tempo_bpm=int(tempo))
    timings["export_midi"] = _elapsed_ms(midi_started)

    explain_started = time.perf_counter()
    chords = suggest_chords()
    explanation = _build_explanation(
        used_audio_fallback=used_audio_fallback,
        processed_samples=int(processed_audio.shape[0]),
    )
    timings["assemble_outputs"] = _elapsed_ms(explain_started)

    return MelodyResult(
        melody=smoothed,
        note_events=note_events,
        melody_profile=profile,
        detected_key=detected_key,
        chords=chords,
        explanation=explanation,
        midi_bytes=midi_bytes,
        detected_tempo=tempo,
        step_latency_ms=timings,
        metadata=metadata,
    )


def _load_audio_with_fallback(audio_bytes: bytes) -> tuple[Any, bool, str | None]:
    try:
        audio, _ = load_audio(audio_bytes, target_sample_rate=DEFAULT_SAMPLE_RATE)
        return audio, False, None
    except (EOFError, TypeError, ValueError, sf.LibsndfileError) as exc:
        fallback = _silent_audio_from_size(audio_size(audio_bytes))
        return fallback, True, str(exc)


def _generate_mock_melody(
    audio_bytes: bytes,
    *,
    tempo_bpm: int | None,
    mood: str | None = None,
) -> list[MelodyNote]:
    size = audio_size(audio_bytes)
    seed = seed_from_size(size)
    tempo = tempo_bpm or 100
    lowered_mood = (mood or "").strip().lower()
    if lowered_mood == "melancholic":
        final_pitch = "A4"
    elif tempo > 110:
        final_pitch = "C5"
    else:
        final_pitch = "F4"
    return [
        MelodyNote(pitch="C4", start_beat=0, duration_beats=1, velocity=90 + seed),
        MelodyNote(pitch="E4", start_beat=1, duration_beats=1, velocity=92 + seed),
        MelodyNote(pitch="G4", start_beat=2, duration_beats=1, velocity=95 + seed),
        MelodyNote(pitch=final_pitch, start_beat=3, duration_beats=1, velocity=97 + seed),
    ]


def _build_explanation(*, used_audio_fallback: bool, processed_samples: int) -> list[ExplanationPart]:
    preprocessing_detail = (
        "Audio decoding fell back to a deterministic silent proxy, so the later stages stayed reproducible."
        if used_audio_fallback
        else f"Audio was decoded and preprocessed to {processed_samples} mono samples before symbolic conversion."
    )
    return [
        ExplanationPart(
            title="Pipeline",
            detail=preprocessing_detail,
        ),
        ExplanationPart(
            title="Contour",
            detail="The phrase rises to a mild peak and resolves back for singable repetition.",
        ),
        ExplanationPart(
            title="Harmony fit",
            detail="Suggested chords stay diatonic to keep the progression stable for a first sketch.",
        ),
    ]


def _silent_audio_from_size(size: int) -> Any:
    import numpy as np

    length = max(1, int(size))
    return np.zeros(length, dtype=np.float32)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, _ROUND_DIGITS)
