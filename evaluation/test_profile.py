"""Tests for melody profile analysis helpers."""

from __future__ import annotations

from ml.melody_sketchpad.profile import build_melody_profile, classify_contour
from shared.schemas import NoteEvent


def test_build_melody_profile_marks_ascending_contour_as_rising() -> None:
    notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.9),
        NoteEvent(pitch=62, onset=1.0, duration=1.0, velocity=98, confidence=0.9),
        NoteEvent(pitch=65, onset=2.0, duration=1.0, velocity=100, confidence=0.9),
    ]

    profile = build_melody_profile(notes)

    assert profile.contour == "rising"
    assert profile.pitch_range == (60, 65)
    assert profile.rhythmic_density == 1.0
    assert profile.interval_histogram == [0.0, 0.0, 0.0, 0.5, 0.5]


def test_classify_contour_identifies_arch_shape() -> None:
    assert classify_contour([60, 67, 62]) == "arch"


def test_classify_contour_identifies_valley_shape() -> None:
    assert classify_contour([67, 60, 65]) == "valley"


def test_build_melody_profile_handles_empty_input() -> None:
    profile = build_melody_profile([])

    assert profile.interval_histogram == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert profile.rhythmic_density == 0.0
    assert profile.pitch_range == (0, 0)
    assert profile.contour == "rising"
