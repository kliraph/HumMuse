"""Preprocessing helpers for hummed audio input."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_FRAME_MS = 30
DEFAULT_HOP_MS = 10
DEFAULT_EDGE_PAD_MS = 50
DEFAULT_THRESHOLD_FLOOR_DBFS = -50.0
DEFAULT_THRESHOLD_DROP_DB = 25.0
DEFAULT_NOISE_REDUCTION_STRENGTH = 0.8
_EPSILON = 1e-12


def _coerce_audio_source(audio_source: bytes | bytearray | str | Path | BinaryIO) -> str | BytesIO | BinaryIO:
    if isinstance(audio_source, (bytes, bytearray)):
        return BytesIO(audio_source)
    if isinstance(audio_source, (str, Path)):
        return str(audio_source)
    return audio_source


def _read_audio(audio_source: bytes | bytearray | str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    source = _coerce_audio_source(audio_source)
    if hasattr(source, "seek"):
        source.seek(0)

    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _resample_audio(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(source_sample_rate)
    target_length = max(1, int(round(duration * target_sample_rate)))
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32, copy=False)


def load_audio(
    audio_source: bytes | bytearray | str | Path | BinaryIO,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Load audio and return mono float32 samples at the target sample rate."""
    audio, sample_rate = _read_audio(audio_source)
    audio = _resample_audio(audio, sample_rate, target_sample_rate)
    return np.clip(audio, -1.0, 1.0), target_sample_rate


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = DEFAULT_FRAME_MS,
    hop_ms: int = DEFAULT_HOP_MS,
    edge_pad_ms: int = DEFAULT_EDGE_PAD_MS,
    adaptive_threshold_floor_dbfs: float = DEFAULT_THRESHOLD_FLOOR_DBFS,
    relative_threshold_drop_db: float = DEFAULT_THRESHOLD_DROP_DB,
) -> np.ndarray:
    """Trim leading and trailing silence while preserving internal pauses."""
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    waveform = audio.astype(np.float32, copy=False)
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_size = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    edge_pad = max(0, int(round(sample_rate * edge_pad_ms / 1000.0)))

    frame_starts = list(range(0, waveform.shape[0], hop_size))
    frame_rms_db = np.empty(len(frame_starts), dtype=np.float32)
    for index, start in enumerate(frame_starts):
        frame = waveform[start : start + frame_size]
        if frame.size == 0:
            frame_rms_db[index] = -np.inf
            continue
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        frame_rms_db[index] = float(20.0 * np.log10(max(rms, _EPSILON)))

    percentile_95 = float(np.percentile(frame_rms_db, 95))
    threshold_db = max(percentile_95 - relative_threshold_drop_db, adaptive_threshold_floor_dbfs)
    active_indices = np.flatnonzero(frame_rms_db >= threshold_db)
    if active_indices.size == 0:
        return waveform

    start_sample = max(0, frame_starts[int(active_indices[0])] - edge_pad)
    end_sample = min(
        waveform.shape[0],
        frame_starts[int(active_indices[-1])] + frame_size + edge_pad,
    )
    return waveform[start_sample:end_sample]


def normalise_lufs(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
) -> np.ndarray:
    """Normalize waveform loudness to the target LUFS when a stable reading is available."""
    waveform = audio.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform
    if waveform.shape[0] / float(sample_rate) < 0.4:
        return waveform

    try:
        meter = pyln.Meter(sample_rate)
        loudness = float(meter.integrated_loudness(waveform.astype(np.float64)))
    except Exception:
        return waveform

    if not np.isfinite(loudness):
        return waveform

    try:
        normalized = pyln.normalize.loudness(waveform.astype(np.float64), loudness, target_lufs)
    except Exception:
        return waveform

    if not np.all(np.isfinite(normalized)):
        return waveform

    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    *,
    enabled: bool = True,
    prop_decrease: float = DEFAULT_NOISE_REDUCTION_STRENGTH,
) -> np.ndarray:
    """Optionally reduce stationary background noise using noisereduce."""
    waveform = audio.astype(np.float32, copy=False)
    if waveform.size == 0 or not enabled:
        return waveform

    try:
        reduced = nr.reduce_noise(
            y=waveform.astype(np.float64),
            sr=sample_rate,
            stationary=True,
            prop_decrease=prop_decrease,
        )
    except Exception:
        return waveform

    if not np.all(np.isfinite(reduced)):
        return waveform

    return np.clip(reduced, -1.0, 1.0).astype(np.float32, copy=False)


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply task 3.3 preprocessing stages after decoding."""
    denoised = reduce_noise(audio, sample_rate)
    trimmed = trim_silence(denoised, sample_rate)
    return normalise_lufs(trimmed, sample_rate)


def audio_size(audio_bytes: bytes) -> int:
    """Return deterministic size feature used by the existing mock pipeline."""
    try:
        audio, _ = load_audio(audio_bytes)
    except (EOFError, TypeError, ValueError, sf.LibsndfileError):
        return len(audio_bytes)
    processed = preprocess_audio(audio, DEFAULT_SAMPLE_RATE)
    return int(processed.shape[0])
