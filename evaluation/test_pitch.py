"""Tests for PESTO-based melody extraction."""

from __future__ import annotations

import numpy as np

import ml.melody_sketchpad.pitch as pitch_module
from shared.schemas import NoteEvent


def _frames_for(midi_pitches: list[float], frames_per_note: int, confidence: float = 0.9):
    """Build PESTO-style (pitch_hz, confidence, hop_s) arrays from a list of MIDI pitches."""
    pitch_hz = []
    conf = []
    for midi in midi_pitches:
        hz = 440.0 * (2 ** ((float(midi) - 69) / 12.0)) if np.isfinite(midi) else 0.0
        c = confidence if np.isfinite(midi) else 0.0
        for _ in range(frames_per_note):
            pitch_hz.append(hz)
            conf.append(c)
    return np.asarray(pitch_hz, dtype=np.float64), np.asarray(conf, dtype=np.float64), 0.010


def test_extract_melody_segments_two_note_phrase(monkeypatch) -> None:
    hz, conf, hop_s = _frames_for([69.0, 71.0], frames_per_note=20, confidence=0.85)
    monkeypatch.setattr(pitch_module, "_pesto_extract_frames", lambda *a, **kw: (hz, conf, hop_s))

    events = pitch_module.extract_melody(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)

    assert len(events) == 2
    assert events[0].pitch == 69
    assert events[0].onset == 0.0
    assert round(events[0].duration, 2) == 0.20
    assert events[0].confidence > 0.8
    assert events[1].pitch == 71
    assert round(events[1].onset, 2) == 0.20


def test_extract_melody_drops_segments_below_minimum_length(monkeypatch) -> None:
    # 50ms of 69, 50ms of 71 — both below the 80ms minimum
    hz, conf, hop_s = _frames_for([69.0, 71.0], frames_per_note=5, confidence=0.9)
    monkeypatch.setattr(pitch_module, "_pesto_extract_frames", lambda *a, **kw: (hz, conf, hop_s))

    events = pitch_module.extract_melody(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)

    assert events == []


def test_extract_melody_drops_unvoiced_frames(monkeypatch) -> None:
    # Voiced 69, unvoiced gap, voiced 71. Builder uses NaN to mark unvoiced.
    hz, conf, hop_s = _frames_for([69.0, float("nan"), 71.0], frames_per_note=15)
    monkeypatch.setattr(pitch_module, "_pesto_extract_frames", lambda *a, **kw: (hz, conf, hop_s))

    events = pitch_module.extract_melody(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)

    assert len(events) == 2
    assert events[0].pitch == 69
    assert events[1].pitch == 71
    # Unvoiced 150ms gap between them
    assert round(events[1].onset - (events[0].onset + events[0].duration), 2) == 0.15


def test_extract_melody_gates_out_of_band_frequencies(monkeypatch) -> None:
    # Voiced frames at 50 Hz (below 80 Hz floor) should be discarded
    hz = np.full(30, 50.0, dtype=np.float64)
    conf = np.full(30, 0.9, dtype=np.float64)
    monkeypatch.setattr(pitch_module, "_pesto_extract_frames", lambda *a, **kw: (hz, conf, 0.010))

    events = pitch_module.extract_melody(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)

    assert events == []


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
