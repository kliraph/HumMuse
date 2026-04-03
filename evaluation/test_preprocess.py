"""Tests for melody audio preprocessing."""

from __future__ import annotations

import wave
from io import BytesIO

import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from ml.melody_sketchpad.preprocess import (
    DEFAULT_EDGE_PAD_MS,
    DEFAULT_NOISE_REDUCTION_STRENGTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TARGET_LUFS,
    load_audio,
    normalise_lufs,
    reduce_noise,
    trim_silence,
)


def _build_stereo_wav_bytes(*, sample_rate: int = 8_000, duration_seconds: float = 1.0) -> bytes:
    timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    left = 0.6 * np.sin(2 * np.pi * 220.0 * timeline)
    right = 0.3 * np.sin(2 * np.pi * 440.0 * timeline)
    stereo = np.column_stack((left, right))
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_load_audio_returns_16khz_mono_float32_array() -> None:
    audio_bytes = _build_stereo_wav_bytes()

    waveform, sample_rate = load_audio(audio_bytes)

    assert sample_rate == DEFAULT_SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.shape == (DEFAULT_SAMPLE_RATE,)
    assert np.max(np.abs(waveform)) <= 1.0


def test_load_audio_reads_mp3_and_resamples_to_16khz_mono() -> None:
    timeline = np.linspace(0.0, 1.0, 22_050, endpoint=False)
    stereo = np.column_stack(
        (
            0.5 * np.sin(2 * np.pi * 330.0 * timeline),
            0.25 * np.sin(2 * np.pi * 660.0 * timeline),
        )
    ).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, stereo, 22_050, format="MP3")

    waveform, sample_rate = load_audio(buffer.getvalue())

    assert sample_rate == DEFAULT_SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.shape == (DEFAULT_SAMPLE_RATE,)
    assert np.max(np.abs(waveform)) <= 1.0


def test_trim_silence_removes_edges_but_preserves_internal_gap() -> None:
    leading = np.zeros(int(0.2 * DEFAULT_SAMPLE_RATE), dtype=np.float32)
    first_phrase = 0.45 * np.sin(2 * np.pi * 220.0 * np.linspace(0.0, 0.35, int(0.35 * DEFAULT_SAMPLE_RATE), endpoint=False))
    internal_gap = np.zeros(int(0.12 * DEFAULT_SAMPLE_RATE), dtype=np.float32)
    second_phrase = 0.4 * np.sin(2 * np.pi * 330.0 * np.linspace(0.0, 0.30, int(0.30 * DEFAULT_SAMPLE_RATE), endpoint=False))
    trailing = np.zeros(int(0.25 * DEFAULT_SAMPLE_RATE), dtype=np.float32)
    waveform = np.concatenate((leading, first_phrase, internal_gap, second_phrase, trailing)).astype(np.float32)

    trimmed = trim_silence(waveform, DEFAULT_SAMPLE_RATE)
    pad_samples = int(DEFAULT_EDGE_PAD_MS * DEFAULT_SAMPLE_RATE / 1000)
    middle_window = trimmed[trimmed.shape[0] // 3 : (2 * trimmed.shape[0]) // 3]
    quiet_mask = np.abs(middle_window) < 0.02
    quiet_runs = np.split(quiet_mask, np.where(np.diff(quiet_mask.astype(np.int8)) != 0)[0] + 1)
    longest_quiet_run = max((run.size for run in quiet_runs if run.size and run[0]), default=0)

    assert trimmed.shape[0] < waveform.shape[0]
    assert trimmed.shape[0] > first_phrase.shape[0] + second_phrase.shape[0]
    assert np.max(np.abs(trimmed[:pad_samples])) < 0.001
    assert np.max(np.abs(trimmed[pad_samples : pad_samples + int(0.05 * DEFAULT_SAMPLE_RATE)])) > 0.01
    assert np.max(np.abs(trimmed[-pad_samples:])) < 0.001
    assert np.max(np.abs(trimmed[-(pad_samples + int(0.05 * DEFAULT_SAMPLE_RATE)) : -pad_samples])) > 0.01
    assert longest_quiet_run >= int(0.08 * DEFAULT_SAMPLE_RATE)


def test_trim_silence_returns_all_zero_input_unchanged() -> None:
    waveform = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)

    trimmed = trim_silence(waveform, DEFAULT_SAMPLE_RATE)

    assert trimmed.shape == waveform.shape
    assert np.allclose(trimmed, waveform)


def test_normalise_lufs_matches_target_within_half_lu() -> None:
    waveform = (
        0.08 * np.sin(2 * np.pi * 220.0 * np.linspace(0.0, 2.0, 2 * DEFAULT_SAMPLE_RATE, endpoint=False))
    ).astype(np.float32)

    normalized = normalise_lufs(waveform, DEFAULT_SAMPLE_RATE)
    loudness = pyln.Meter(DEFAULT_SAMPLE_RATE).integrated_loudness(normalized.astype(np.float64))

    assert np.isfinite(loudness)
    assert abs(loudness - DEFAULT_TARGET_LUFS) <= 0.5


def test_normalise_lufs_keeps_short_clip_finite_without_error() -> None:
    waveform = (
        0.2 * np.sin(2 * np.pi * 440.0 * np.linspace(0.0, 0.15, int(0.15 * DEFAULT_SAMPLE_RATE), endpoint=False))
    ).astype(np.float32)

    normalized = normalise_lufs(waveform, DEFAULT_SAMPLE_RATE)

    assert normalized.dtype == np.float32
    assert normalized.shape == waveform.shape
    assert np.all(np.isfinite(normalized))


def test_reduce_noise_makes_stationary_noise_quieter() -> None:
    rng = np.random.default_rng(7)
    noise_only = 0.04 * rng.standard_normal(int(0.3 * DEFAULT_SAMPLE_RATE))
    voiced = (
        0.22 * np.sin(2 * np.pi * 220.0 * np.linspace(0.0, 1.0, DEFAULT_SAMPLE_RATE, endpoint=False))
        + 0.04 * rng.standard_normal(DEFAULT_SAMPLE_RATE)
    )
    waveform = np.concatenate((noise_only, voiced)).astype(np.float32)

    reduced = reduce_noise(waveform, DEFAULT_SAMPLE_RATE)

    before_noise_floor = float(np.mean(np.square(waveform[: noise_only.shape[0]], dtype=np.float64)))
    after_noise_floor = float(np.mean(np.square(reduced[: noise_only.shape[0]], dtype=np.float64)))

    assert nr is not None
    assert reduced.dtype == np.float32
    assert reduced.shape == waveform.shape
    assert np.all(np.isfinite(reduced))
    assert after_noise_floor < before_noise_floor


def test_preprocess_audio_runs_denoise_then_trim_then_normalize(monkeypatch) -> None:
    from ml.melody_sketchpad import preprocess as preprocess_module

    calls: list[str] = []
    waveform = np.array([0.1, 0.2], dtype=np.float32)

    def fake_reduce_noise(audio, sample_rate):
        calls.append("reduce_noise")
        assert sample_rate == DEFAULT_SAMPLE_RATE
        assert np.array_equal(audio, waveform)
        return np.array([0.3, 0.4], dtype=np.float32)

    def fake_trim_silence(audio, sample_rate):
        calls.append("trim_silence")
        assert sample_rate == DEFAULT_SAMPLE_RATE
        assert np.array_equal(audio, np.array([0.3, 0.4], dtype=np.float32))
        return np.array([0.5], dtype=np.float32)

    def fake_normalise_lufs(audio, sample_rate):
        calls.append("normalise_lufs")
        assert sample_rate == DEFAULT_SAMPLE_RATE
        assert np.array_equal(audio, np.array([0.5], dtype=np.float32))
        return np.array([0.6], dtype=np.float32)

    monkeypatch.setattr(preprocess_module, "reduce_noise", fake_reduce_noise)
    monkeypatch.setattr(preprocess_module, "trim_silence", fake_trim_silence)
    monkeypatch.setattr(preprocess_module, "normalise_lufs", fake_normalise_lufs)

    processed = preprocess_module.preprocess_audio(waveform, DEFAULT_SAMPLE_RATE)

    assert calls == ["reduce_noise", "trim_silence", "normalise_lufs"]
    assert np.array_equal(processed, np.array([0.6], dtype=np.float32))
