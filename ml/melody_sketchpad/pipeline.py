"""End-to-end melody pipeline with per-step latency logging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import soundfile as sf

from ml.melody_sketchpad.key_detect import detect_key
from ml.melody_sketchpad.midi_export import notes_to_midi
from ml.melody_sketchpad.notes import melody_notes_to_events, note_events_to_melody_notes
from ml.melody_sketchpad.pitch import extract_melody_or_fallback, seed_from_size
from ml.melody_sketchpad.preprocess import (
    DEFAULT_SAMPLE_RATE,
    audio_size,
    load_audio,
    preprocess_audio,
)
from ml.melody_sketchpad.profile import build_melody_profile
from ml.melody_sketchpad.smooth import smooth_notes
from shared.schemas import ExplanationPart, MelodyNote, MelodyProfile, NoteEvent

_ROUND_DIGITS = 3


@dataclass(frozen=True)
class MelodyResult:
    melody: list[MelodyNote]
    note_events: list[NoteEvent]
    melody_profile: MelodyProfile
    detected_key: str | None
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
    """Execute the melody pipeline from audio bytes to a structured result.

    Tempo resolution policy (recorded in ``metadata['tempo_source']``):
      - ``user_provided``: caller passed a value → it wins, no detection.
      - ``fallback_silent_audio``: decode fell back to a silent proxy
        (invalid/unreadable upload) → 100 BPM, can't detect from silence.
      - ``librosa_beat_track``: librosa estimated tempo from the
        preprocessed audio.
      - ``fallback_detection_failed``: librosa raised or returned a
        non-finite value → 100 BPM safety net.
    """
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    decode_started = time.perf_counter()
    audio, used_audio_fallback, decode_error = _load_audio_with_fallback(audio_bytes)
    timings["decode_audio"] = _elapsed_ms(decode_started)
    metadata["used_audio_fallback"] = used_audio_fallback
    if decode_error is not None:
        metadata["decode_error"] = decode_error

    preprocess_started = time.perf_counter()
    processed_audio = preprocess_audio(audio, DEFAULT_SAMPLE_RATE)
    timings["preprocess_audio"] = _elapsed_ms(preprocess_started)

    tempo_started = time.perf_counter()
    detection_succeeded = True
    if tempo_bpm is not None:
        # Manual override — caller knows the tempo (or wants to force it
        # for reproducibility in tests/golden runs). Skip detection.
        tempo = float(tempo_bpm)
        metadata["tempo_source"] = "user_provided"
    elif used_audio_fallback:
        # Detecting tempo from a silent placeholder would be noise.
        tempo = 100.0
        metadata["tempo_source"] = "fallback_silent_audio"
    else:
        tempo, detection_succeeded = _detect_tempo(processed_audio, DEFAULT_SAMPLE_RATE)
        metadata["tempo_source"] = (
            "librosa_beat_track" if detection_succeeded else "fallback_detection_failed"
        )
    metadata["tempo_bpm"] = round(tempo, 2)
    timings["detect_tempo"] = _elapsed_ms(tempo_started)
    # Downstream stages expect an int tempo hint; pass the resolved
    # value so quantization and MIDI export agree with detected_tempo.
    resolved_tempo_int = int(round(tempo))

    notes_started = time.perf_counter()
    # Only let the IOI estimator override when librosa's beat tracker fell back
    # to its flat 100 BPM net; a user-provided or successfully-detected tempo
    # always wins.
    allow_ioi_fallback = (
        tempo_bpm is None and not used_audio_fallback and not detection_succeeded
    )
    melody, extraction_meta, resolved_tempo = _extract_melody_notes(
        processed_audio,
        DEFAULT_SAMPLE_RATE,
        audio_bytes=audio_bytes,
        tempo_bpm=resolved_tempo_int,
        mood=mood,
        used_audio_fallback=used_audio_fallback,
        allow_ioi_fallback=allow_ioi_fallback,
    )
    timings["generate_notes"] = _elapsed_ms(notes_started)
    metadata.update(extraction_meta)
    # The IOI fallback may have corrected the tempo from the flat detection net;
    # keep detected_tempo, metadata, and MIDI export aligned with the beats the
    # notes were actually converted into.
    tempo = float(resolved_tempo)
    resolved_tempo_int = int(round(tempo))
    metadata["tempo_bpm"] = round(tempo, 2)

    # Quantization is intentionally NOT applied here. PESTO produces 10 ms-
    # resolution onsets that respect the natural rubato of humming; snapping
    # to an 1/8 grid at typical hum tempos (60-140 BPM) crushes distinct
    # onsets into the same cell and produces phantom polyphony at MIDI export.
    # Quantization stays available as a user-facing post-step via
    # ml.melody_sketchpad.quantize.quantize_notes if a snapped grid is wanted.

    smooth_started = time.perf_counter()
    smoothed = smooth_notes(melody)
    timings["smooth_notes"] = _elapsed_ms(smooth_started)

    events_started = time.perf_counter()
    note_events = melody_notes_to_events(smoothed)
    profile = build_melody_profile(note_events)
    timings["build_profile"] = _elapsed_ms(events_started)

    key_started = time.perf_counter()
    detected_key = detect_key(note_events)
    timings["detect_key"] = _elapsed_ms(key_started)

    midi_started = time.perf_counter()
    midi_bytes = notes_to_midi(smoothed, tempo_bpm=resolved_tempo_int)
    timings["export_midi"] = _elapsed_ms(midi_started)

    explain_started = time.perf_counter()
    explanation = _build_explanation(
        used_audio_fallback=used_audio_fallback,
        processed_samples=int(processed_audio.shape[0]),
        melody_source=str(metadata.get("melody_source", "unknown")),
        raw_note_count=int(metadata.get("raw_note_count", 0)),
    )
    timings["assemble_outputs"] = _elapsed_ms(explain_started)

    return MelodyResult(
        melody=smoothed,
        note_events=note_events,
        melody_profile=profile,
        detected_key=detected_key,
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


def _detect_tempo(audio: Any, sample_rate: int) -> tuple[float, bool]:
    """Estimate tempo from preprocessed audio via librosa beat tracking.

    Returns ``(tempo_bpm, succeeded)``. On any failure — librosa import
    error, internal raise, non-finite/non-positive result, empty audio —
    returns ``(100.0, False)`` so callers always have a usable scalar.
    The boolean lets the pipeline record whether the value came from
    real detection or the safety net (see ``metadata['tempo_source']``).

    100 BPM matches the historical hardcoded default the pipeline used
    before detection existed, so failure modes don't change behaviour
    versus pre-detection runs.
    """
    import numpy as np

    try:
        import librosa

        tempo_array, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        tempo_value = float(np.atleast_1d(tempo_array)[0])
    except Exception:
        return 100.0, False

    if not np.isfinite(tempo_value) or tempo_value <= 0:
        return 100.0, False
    return tempo_value, True


def estimate_tempo_from_onsets(
    onsets_seconds: list[float],
    *,
    minimum_bpm: float = 60.0,
    maximum_bpm: float = 160.0,
) -> float | None:
    """Median inter-onset-interval tempo estimate, octave-folded into range.

    Used as a fallback when librosa beat tracking fails on short hums (it tends
    to return the flat 100 BPM safety net, which misaligns the seconds->beats
    conversion and the downstream harmony grid). Assumes the median gap between
    consecutive onsets is roughly one beat, then doubles/halves the result until
    it lands in ``[minimum_bpm, maximum_bpm)``.

    Returns ``None`` when there are too few onsets to estimate, so callers can
    keep their existing fallback value.
    """
    import numpy as np

    onsets = sorted(float(value) for value in onsets_seconds)
    if len(onsets) < 2:
        return None
    intervals = np.diff(onsets)
    intervals = intervals[intervals > 1e-3]
    if intervals.size == 0:
        return None
    beat_seconds = float(np.median(intervals))
    if beat_seconds <= 0:
        return None
    bpm = 60.0 / beat_seconds
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    while bpm < minimum_bpm:
        bpm *= 2.0
    while bpm >= maximum_bpm:
        bpm /= 2.0
    return bpm


class NoMelodyDetectedError(ValueError):
    """Raised when the extractor finds no melodic content in the audio."""


def _extract_melody_notes(
    audio: Any,
    sample_rate: int,
    *,
    audio_bytes: bytes,
    tempo_bpm: int | None,
    mood: str | None,
    used_audio_fallback: bool,
    allow_ioi_fallback: bool = False,
) -> tuple[list[MelodyNote], dict[str, Any], float]:
    """Extract beat-based melody notes, returning the tempo actually used.

    When ``allow_ioi_fallback`` is set (librosa beat tracking failed and the
    caller did not supply a tempo), the raw second-valued PESTO onsets feed
    :func:`estimate_tempo_from_onsets`, and the resulting tempo replaces the
    flat detection fallback for the seconds->beats conversion. The resolved
    tempo is returned so the caller can keep ``detected_tempo`` and MIDI export
    consistent with the beat positions.
    """
    meta: dict[str, Any] = {}
    resolved_tempo = float(tempo_bpm if tempo_bpm else 100)
    if used_audio_fallback:
        meta["melody_source"] = "mock_decode_fallback"
        return _generate_mock_melody(audio_bytes, tempo_bpm=tempo_bpm, mood=mood), meta, resolved_tempo

    confidence_result = extract_melody_or_fallback(audio, sample_rate)
    meta["pitched_ratio"] = round(float(confidence_result.pitched_ratio), 4)
    meta["raw_note_count"] = len(confidence_result.notes)

    if not confidence_result.notes:
        raise NoMelodyDetectedError(
            "No melody detected in the uploaded audio. Try recording a clearer hum, "
            "closer to the microphone, with sustained pitched notes."
        )

    meta["melody_source"] = (
        "pesto_rhythm_fallback" if confidence_result.used_fallback else "pesto"
    )

    if allow_ioi_fallback:
        ioi_tempo = estimate_tempo_from_onsets(
            [float(note.onset) for note in confidence_result.notes]
        )
        if ioi_tempo is not None:
            resolved_tempo = ioi_tempo
            meta["tempo_source"] = "ioi_fallback"
            meta["tempo_bpm"] = round(ioi_tempo, 2)

    melody = note_events_to_melody_notes(
        confidence_result.notes, tempo_bpm=int(round(resolved_tempo))
    )
    return melody, meta, resolved_tempo


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


def _build_explanation(
    *,
    used_audio_fallback: bool,
    processed_samples: int,
    melody_source: str,
    raw_note_count: int,
) -> list[ExplanationPart]:
    if used_audio_fallback:
        preprocessing_detail = (
            "Audio decoding fell back to a deterministic silent proxy, so the later stages stayed reproducible."
        )
    elif melody_source == "pesto":
        preprocessing_detail = (
            f"Audio was decoded to {processed_samples} mono samples and PESTO extracted "
            f"{raw_note_count} note events before quantize and smoothing."
        )
    elif melody_source == "pesto_rhythm_fallback":
        preprocessing_detail = (
            f"Audio was decoded to {processed_samples} mono samples; pitch confidence was low, "
            "so rhythm-only onsets were used."
        )
    else:
        preprocessing_detail = (
            f"Audio was decoded to {processed_samples} mono samples (melody_source={melody_source})."
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
