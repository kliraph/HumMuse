"""Tests for continuation primer MIDI adaptation."""

from __future__ import annotations

from pathlib import Path

from ml.melody_sketchpad.continuation.primer import notes_to_prompt_midi
from ml.melody_sketchpad.continuation.tokenizer_setup import build_tokenizer
from shared.schemas import NoteEvent


def test_notes_to_prompt_midi_round_trips_through_remi_preserving_pitch_order(tmp_path: Path) -> None:
    notes = [
        NoteEvent(pitch=67, onset=0.95, duration=0.25, velocity=96, confidence=0.9),
        NoteEvent(pitch=60, onset=0.10, duration=0.40, velocity=92, confidence=0.9),
        NoteEvent(pitch=64, onset=0.55, duration=0.30, velocity=94, confidence=0.9),
        NoteEvent(pitch=72, onset=1.30, duration=0.35, velocity=98, confidence=0.9),
    ]
    expected_pitches = [60, 64, 67, 72]

    midi_path = notes_to_prompt_midi(notes, tempo=120.0, out_path=tmp_path / "primer.mid")
    tokenizer = build_tokenizer()

    encoded = tokenizer(midi_path)
    sequence = encoded[0] if isinstance(encoded, list) else encoded
    decoded = tokenizer.decode([sequence])
    decoded_notes = sorted(decoded.tracks[0].notes, key=lambda note: (note.start, note.pitch))

    assert midi_path.exists()
    assert [int(note.pitch) for note in decoded_notes] == expected_pitches
    assert [int(note.start) for note in decoded_notes] == sorted(int(note.start) for note in decoded_notes)


def test_notes_to_prompt_midi_rejects_nonpositive_tempo(tmp_path: Path) -> None:
    notes = [NoteEvent(pitch=60, onset=0.0, duration=0.5, velocity=96, confidence=0.9)]

    try:
        notes_to_prompt_midi(notes, tempo=0.0, out_path=tmp_path / "bad.mid")
    except ValueError as exc:
        assert "tempo must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
