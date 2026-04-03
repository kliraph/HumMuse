"""Tests for Basic Pitch melody extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ml.melody_sketchpad.pitch as pitch_module
from shared.schemas import NoteEvent


class _PredictRecorder:
    def __init__(self, note_events: list[tuple[float, float, int, float, list[int] | None]]) -> None:
        self.note_events = note_events
        self.calls: list[dict[str, object]] = []

    def predict(self, audio_path: str, **kwargs):
        self.calls.append({"audio_path": audio_path, **kwargs})
        return {}, None, self.note_events


def test_extract_melody_maps_basic_pitch_events_to_note_events(monkeypatch) -> None:
    recorder = _PredictRecorder(
        [
            (0.15, 0.55, 69, 0.82, None),
            (0.60, 0.92, 71, 0.31, None),
        ]
    )
    monkeypatch.setattr(pitch_module, "basic_pitch_inference", recorder)
    monkeypatch.setattr(pitch_module, "ICASSP_2022_MODEL_PATH", "mock-model")
    monkeypatch.setattr(pitch_module, "_resolve_model_path", lambda: "resolved-model")

    events = pitch_module.extract_melody(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)

    assert len(events) == 2
    assert events[0].pitch == 69
    assert events[0].onset == 0.15
    assert round(events[0].duration, 2) == 0.40
    assert events[0].confidence == 0.82
    assert events[0].velocity == 104
    assert events[1].pitch == 71


def test_extract_melody_calls_basic_pitch_with_phase_3_4_config(monkeypatch) -> None:
    recorder = _PredictRecorder([(0.0, 0.1, 69, 0.5, None)])
    monkeypatch.setattr(pitch_module, "basic_pitch_inference", recorder)
    monkeypatch.setattr(pitch_module, "ICASSP_2022_MODEL_PATH", "mock-model")
    monkeypatch.setattr(pitch_module, "_resolve_model_path", lambda: "resolved-model")

    events = pitch_module.extract_melody(np.ones(8_000, dtype=np.float32) * 0.1, 16_000)

    assert len(events) == 1
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["model_or_model_path"] == "resolved-model"
    assert call["minimum_frequency"] == pitch_module.DEFAULT_MINIMUM_FREQUENCY
    assert call["maximum_frequency"] == pitch_module.DEFAULT_MAXIMUM_FREQUENCY
    assert call["minimum_note_length"] == pitch_module.DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000.0
    assert Path(str(call["audio_path"])).suffix == ".wav"


def test_extract_melody_returns_empty_list_for_empty_audio() -> None:
    events = pitch_module.extract_melody(np.array([], dtype=np.float32), 16_000)

    assert events == []


def test_calculate_pitched_ratio_merges_overlapping_note_coverage() -> None:
    notes = [
        NoteEvent(pitch=69, onset=0.0, duration=0.4, velocity=100, confidence=0.9),
        NoteEvent(pitch=71, onset=0.3, duration=0.4, velocity=95, confidence=0.85),
        NoteEvent(pitch=72, onset=0.8, duration=0.1, velocity=90, confidence=0.8),
    ]

    ratio = pitch_module.calculate_pitched_ratio(notes, 1.0)

    assert round(ratio, 2) == 0.8


def test_analyse_pitch_confidence_keeps_clean_humming_above_threshold() -> None:
    notes = [
        NoteEvent(pitch=69, onset=0.0, duration=0.35, velocity=100, confidence=0.9),
        NoteEvent(pitch=71, onset=0.35, duration=0.35, velocity=98, confidence=0.88),
        NoteEvent(pitch=72, onset=0.7, duration=0.12, velocity=96, confidence=0.86),
    ]

    result = pitch_module.analyse_pitch_confidence(notes, 1.0)

    assert result.pitched_ratio > 0.7
    assert result.should_trigger_fallback is False


def test_analyse_pitch_confidence_triggers_fallback_for_noisy_clip() -> None:
    notes = [
        NoteEvent(pitch=69, onset=0.1, duration=0.08, velocity=80, confidence=0.42),
        NoteEvent(pitch=70, onset=0.45, duration=0.09, velocity=82, confidence=0.4),
    ]

    result = pitch_module.analyse_pitch_confidence(notes, 1.0)

    assert result.pitched_ratio < pitch_module.DEFAULT_PITCHED_RATIO_THRESHOLD
    assert result.should_trigger_fallback is True


def test_extract_melody_with_confidence_wraps_note_extraction(monkeypatch) -> None:
    notes = [
        NoteEvent(pitch=69, onset=0.0, duration=0.5, velocity=100, confidence=0.9),
        NoteEvent(pitch=71, onset=0.5, duration=0.25, velocity=98, confidence=0.88),
    ]
    monkeypatch.setattr(pitch_module, "extract_melody", lambda *args, **kwargs: notes)

    result = pitch_module.extract_melody_with_confidence(np.ones(16_000, dtype=np.float32), 16_000)

    assert result.notes == notes
    assert result.pitched_ratio == 0.75
    assert result.should_trigger_fallback is False
    assert result.used_fallback is False
    assert result.pitched_notes == notes


def test_detect_rhythm_fallback_returns_rhythm_only_events_for_impulses(monkeypatch) -> None:
    sample_rate = 16_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    for onset_seconds in (0.2, 0.5, 0.8):
        center = int(onset_seconds * sample_rate)
        audio[center : center + 160] = np.hanning(160).astype(np.float32)

    strength_calls: list[dict[str, object]] = []
    detect_calls: list[dict[str, object]] = []

    class _FakeOnset:
        @staticmethod
        def onset_strength(**kwargs):
            strength_calls.append(kwargs)
            envelope = np.zeros(32, dtype=np.float64)
            envelope[[6, 16, 25]] = [0.7, 0.85, 0.9]
            return envelope

        @staticmethod
        def onset_detect(**kwargs):
            detect_calls.append(kwargs)
            return np.array([6, 16, 25], dtype=int)

    class _FakeLibrosa:
        onset = _FakeOnset()

        @staticmethod
        def frames_to_time(frames, sr, hop_length):
            return np.asarray(frames, dtype=np.float64) * hop_length / sr

    monkeypatch.setattr(pitch_module, "librosa", _FakeLibrosa())
    events = pitch_module.detect_rhythm_fallback(audio, sample_rate)

    assert len(strength_calls) == 1
    assert strength_calls[0]["sr"] == sample_rate
    assert strength_calls[0]["hop_length"] == pitch_module.DEFAULT_ONSET_HOP_LENGTH
    expected_envelope = np.zeros(32, dtype=np.float64)
    expected_envelope[[6, 16, 25]] = [0.7, 0.85, 0.9]
    assert np.allclose(detect_calls[0]["onset_envelope"], expected_envelope)
    assert len(events) >= 3
    assert all(event.pitch == pitch_module.DEFAULT_FALLBACK_PITCH for event in events)
    assert all(event.duration > 0 for event in events)
    assert all(0.0 <= event.confidence <= 1.0 for event in events)
    assert abs(events[0].onset - 0.2) < 0.08


def test_extract_melody_or_fallback_uses_rhythm_notes_when_triggered(monkeypatch) -> None:
    pitched_notes = [
        NoteEvent(pitch=69, onset=0.1, duration=0.08, velocity=80, confidence=0.42),
        NoteEvent(pitch=70, onset=0.45, duration=0.09, velocity=82, confidence=0.4),
    ]
    fallback_notes = [
        NoteEvent(pitch=60, onset=0.12, duration=0.3, velocity=70, confidence=0.3),
        NoteEvent(pitch=60, onset=0.55, duration=0.25, velocity=72, confidence=0.32),
    ]
    monkeypatch.setattr(pitch_module, "extract_melody", lambda *args, **kwargs: pitched_notes)
    monkeypatch.setattr(pitch_module, "detect_rhythm_fallback", lambda *args, **kwargs: fallback_notes)

    result = pitch_module.extract_melody_or_fallback(np.ones(16_000, dtype=np.float32), 16_000)

    assert result.should_trigger_fallback is True
    assert result.used_fallback is True
    assert result.notes == fallback_notes
    assert result.pitched_notes == pitched_notes
    assert result.pitched_ratio < pitch_module.DEFAULT_PITCHED_RATIO_THRESHOLD


def test_extract_melody_or_fallback_keeps_pitched_notes_when_confident(monkeypatch) -> None:
    pitched_notes = [
        NoteEvent(pitch=69, onset=0.0, duration=0.5, velocity=100, confidence=0.9),
        NoteEvent(pitch=71, onset=0.5, duration=0.25, velocity=98, confidence=0.88),
    ]
    monkeypatch.setattr(pitch_module, "extract_melody", lambda *args, **kwargs: pitched_notes)

    result = pitch_module.extract_melody_or_fallback(np.ones(16_000, dtype=np.float32), 16_000)

    assert result.should_trigger_fallback is False
    assert result.used_fallback is False
    assert result.notes == pitched_notes
    assert result.pitched_notes == pitched_notes
