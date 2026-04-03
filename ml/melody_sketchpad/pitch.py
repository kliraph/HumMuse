"""Pitch extraction helpers for melody sketchpad."""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.melody_sketchpad.notes import basic_pitch_notes_to_events, confidence_to_velocity
from shared.schemas import NoteEvent

basic_pitch_inference = None
ICASSP_2022_MODEL_PATH = None

try:
    import librosa
except ModuleNotFoundError:  # pragma: no cover - optional dependency until installed in the env
    librosa = None

DEFAULT_MINIMUM_FREQUENCY = 80.0
DEFAULT_MAXIMUM_FREQUENCY = 800.0
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 80.0
DEFAULT_PITCHED_RATIO_THRESHOLD = 0.4
DEFAULT_FALLBACK_PITCH = 60
DEFAULT_ONSET_HOP_LENGTH = 512
DEFAULT_RHYTHM_CONFIDENCE_FLOOR = 0.2


def seed_from_size(size: int) -> int:
    return max(1, min(size // 1000, 4))


def _basic_pitch_is_available() -> bool:
    _ensure_basic_pitch_loaded()
    return basic_pitch_inference is not None and ICASSP_2022_MODEL_PATH is not None


def _ensure_basic_pitch_loaded() -> None:
    global basic_pitch_inference, ICASSP_2022_MODEL_PATH
    if basic_pitch_inference is not None and ICASSP_2022_MODEL_PATH is not None:
        return
    try:
        basic_pitch_module = import_module("basic_pitch")
        basic_pitch_inference = import_module("basic_pitch.inference")
        ICASSP_2022_MODEL_PATH = getattr(basic_pitch_module, "ICASSP_2022_MODEL_PATH")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency until installed in the env
        basic_pitch_inference = None
        ICASSP_2022_MODEL_PATH = None


def _resolve_model_path() -> str | Path:
    if not _basic_pitch_is_available():
        raise RuntimeError("basic_pitch is not available in the current Python environment")

    base_model_path = Path(str(ICASSP_2022_MODEL_PATH))
    onnx_model_path = base_model_path.with_suffix(".onnx")
    if importlib.util.find_spec("onnxruntime") is not None and onnx_model_path.exists():
        # basic_pitch always probes TensorFlow first; flipping these flags steers the loader to ONNX.
        basic_pitch_inference.TF_PRESENT = False
        basic_pitch_inference.TFLITE_PRESENT = False
        return onnx_model_path
    return base_model_path

@dataclass(frozen=True)
class PitchConfidenceResult:
    notes: list[NoteEvent]
    pitched_ratio: float
    should_trigger_fallback: bool
    used_fallback: bool = False
    pitched_notes: list[NoteEvent] = field(default_factory=list)


def calculate_pitched_ratio(note_events: list[NoteEvent], audio_duration: float) -> float:
    """Return the proportion of the clip covered by pitched note events."""
    if audio_duration <= 0 or not note_events:
        return 0.0

    intervals = sorted(
        (
            max(0.0, float(note.onset)),
            max(0.0, min(audio_duration, float(note.onset) + float(note.duration))),
        )
        for note in note_events
        if float(note.duration) > 0
    )
    if not intervals:
        return 0.0

    merged_duration = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if end <= current_start:
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged_duration += max(0.0, current_end - current_start)
        current_start, current_end = start, end

    merged_duration += max(0.0, current_end - current_start)
    return max(0.0, min(1.0, merged_duration / audio_duration))


def analyse_pitch_confidence(
    note_events: list[NoteEvent],
    audio_duration: float,
    *,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    pitched_ratio = calculate_pitched_ratio(note_events, audio_duration)
    return PitchConfidenceResult(
        notes=note_events,
        pitched_ratio=pitched_ratio,
        should_trigger_fallback=pitched_ratio < pitched_ratio_threshold,
        used_fallback=False,
        pitched_notes=note_events,
    )


def detect_rhythm_fallback(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fallback_pitch: int = DEFAULT_FALLBACK_PITCH,
    hop_length: int = DEFAULT_ONSET_HOP_LENGTH,
) -> list[NoteEvent]:
    """Use onset detection to derive rhythm-only note events for noisy clips."""
    waveform = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if waveform.size == 0 or sample_rate <= 0 or librosa is None:
        return []

    try:
        onset_envelope = librosa.onset.onset_strength(
            y=waveform.astype(np.float64),
            sr=sample_rate,
            hop_length=hop_length,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
        )
    except Exception:
        return []

    audio_duration = waveform.shape[0] / float(sample_rate)
    if onset_frames.size == 0:
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if rms <= 1e-4:
            return []
        onset_frames = np.array([0], dtype=int)

    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)
    onset_times = np.clip(onset_times.astype(np.float64, copy=False), 0.0, audio_duration)
    onset_strengths = onset_envelope[onset_frames] if onset_envelope.size else np.array([], dtype=np.float64)
    max_strength = float(np.max(onset_strengths)) if onset_strengths.size else 0.0
    median_interval = (
        float(np.median(np.diff(onset_times)))
        if onset_times.size > 1
        else min(0.25, max(audio_duration, 0.05))
    )

    events: list[NoteEvent] = []
    for index, onset_time in enumerate(onset_times):
        if index + 1 < onset_times.size:
            duration = max(1.0 / sample_rate, float(onset_times[index + 1] - onset_time))
        else:
            remaining = max(1.0 / sample_rate, float(audio_duration - onset_time))
            duration = max(1.0 / sample_rate, min(remaining, median_interval))

        strength = float(onset_strengths[index]) if index < onset_strengths.size else 0.0
        if max_strength > 0:
            confidence = DEFAULT_RHYTHM_CONFIDENCE_FLOOR + (
                (1.0 - DEFAULT_RHYTHM_CONFIDENCE_FLOOR) * (strength / max_strength)
            )
        else:
            confidence = DEFAULT_RHYTHM_CONFIDENCE_FLOOR
        confidence = max(0.0, min(1.0, confidence))

        events.append(
            NoteEvent(
                pitch=fallback_pitch,
                onset=float(onset_time),
                duration=duration,
                velocity=confidence_to_velocity(confidence),
                confidence=confidence,
            )
        )

    return events


def extract_melody(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
) -> list[NoteEvent]:
    """Extract monophonic note events with Basic Pitch."""
    if audio.size == 0:
        return []
    if not _basic_pitch_is_available():
        raise RuntimeError("basic_pitch is not available in the current Python environment")

    waveform = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    model_path = _resolve_model_path()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        minimum_note_length_seconds = max(0.0, float(minimum_note_length_ms)) / 1000.0
        sf.write(temp_path, waveform, sample_rate)
        _, _, note_events = basic_pitch_inference.predict(
            str(temp_path),
            model_or_model_path=model_path,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            minimum_note_length=minimum_note_length_seconds,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return basic_pitch_notes_to_events(note_events)


def extract_melody_with_confidence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    """Extract note events and derive fallback confidence metadata for noisy clips."""
    audio_duration = 0.0 if sample_rate <= 0 else audio.shape[0] / float(sample_rate)
    notes = extract_melody(
        audio,
        sample_rate,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        minimum_note_length_ms=minimum_note_length_ms,
    )
    return analyse_pitch_confidence(
        notes,
        audio_duration,
        pitched_ratio_threshold=pitched_ratio_threshold,
    )


def extract_melody_or_fallback(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    """Return pitched notes when reliable, otherwise rhythm-only fallback notes."""
    confidence_result = extract_melody_with_confidence(
        audio,
        sample_rate,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        minimum_note_length_ms=minimum_note_length_ms,
        pitched_ratio_threshold=pitched_ratio_threshold,
    )
    if not confidence_result.should_trigger_fallback:
        return confidence_result

    fallback_notes = detect_rhythm_fallback(audio, sample_rate)
    return PitchConfidenceResult(
        notes=fallback_notes,
        pitched_ratio=confidence_result.pitched_ratio,
        should_trigger_fallback=True,
        used_fallback=True,
        pitched_notes=confidence_result.notes,
    )
