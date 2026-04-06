"""Tests for the melody sketchpad pipeline."""

from __future__ import annotations

from ml.melody_sketchpad.pipeline import MelodyResult, run_melody_pipeline


def test_run_melody_pipeline_returns_structured_result_for_invalid_wav_bytes() -> None:
    result = run_melody_pipeline(b"RIFF....WAVEfmt", tempo_bpm=120)

    assert isinstance(result, MelodyResult)
    assert len(result.melody) >= 1
    assert len(result.note_events) >= 1
    assert result.melody_profile is not None
    assert result.midi_bytes.startswith(b"MThd")
    assert result.detected_tempo == 120.0
    assert result.metadata["used_audio_fallback"] is True
    assert "decode_audio" in result.step_latency_ms
    assert "preprocess_audio" in result.step_latency_ms
    assert "generate_notes" in result.step_latency_ms
    assert "export_midi" in result.step_latency_ms


def test_run_melody_pipeline_explanation_mentions_preprocessing_path() -> None:
    result = run_melody_pipeline(b"RIFF....WAVEfmt", tempo_bpm=100)

    assert result.explanation[0].title == "Pipeline"
    assert "deterministic silent proxy" in result.explanation[0].detail
