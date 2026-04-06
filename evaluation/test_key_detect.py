"""Tests for melody key detection."""

from __future__ import annotations

from ml.melody_sketchpad.key_detect import detect_key
from ml.melody_sketchpad.pipeline import run_melody_pipeline
from shared.schemas import NoteEvent


def test_detect_key_returns_c_major_for_c_major_melody() -> None:
    notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=72, onset=3.0, duration=1.0, velocity=96, confidence=1.0),
    ]

    assert detect_key(notes) == "C major"


def test_detect_key_returns_a_minor_for_a_minor_melody() -> None:
    notes = [
        NoteEvent(pitch=69, onset=0.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=72, onset=1.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=76, onset=2.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=81, onset=3.0, duration=1.0, velocity=96, confidence=1.0),
    ]

    assert detect_key(notes) == "A minor"


def test_detect_key_returns_none_for_single_pitch_class_input() -> None:
    notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=72, onset=1.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=84, onset=2.0, duration=1.0, velocity=96, confidence=1.0),
    ]

    assert detect_key(notes) is None


def test_run_melody_pipeline_detects_c_major_for_uplift_phrase() -> None:
    result = run_melody_pipeline(b"RIFF....WAVEfmt", tempo_bpm=120, mood="uplift")

    assert result.detected_key == "C major"


def test_run_melody_pipeline_detects_a_minor_for_melancholic_phrase() -> None:
    result = run_melody_pipeline(b"RIFF....WAVEfmt", tempo_bpm=96, mood="melancholic")

    assert result.detected_key == "A minor"
