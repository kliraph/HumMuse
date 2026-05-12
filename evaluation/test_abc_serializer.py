"""Tests for the ABC notation serializer used by the explanation context."""

from __future__ import annotations

from uuid import uuid4

import pytest

from ml.gpt.abc_serializer import (
    ABCSerializationOptions,
    midi_to_abc_pitch,
    session_state_to_abc,
)
from shared.schemas import ChordProgression, NoteEvent, SessionState


def _note(pitch: int, onset: float, duration: float = 1.0, confidence: float = 0.9) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, duration=duration, velocity=100, confidence=confidence)


def _session_state(*, notes: list[NoteEvent] | None = None, chords: list[str] | None = None) -> SessionState:
    return SessionState(
        session_id=uuid4(),
        detected_key="C major",
        detected_tempo=110,
        melody_notes=notes or [],
        chord_progressions=[
            ChordProgression(chords=chords or [], score=0.8, explanation="fixture progression")
        ] if chords else [],
    )


def test_midi_to_abc_pitch_covers_main_octaves() -> None:
    assert midi_to_abc_pitch(60) == "C"
    assert midi_to_abc_pitch(72) == "c"
    assert midi_to_abc_pitch(48) == "C,"
    assert midi_to_abc_pitch(84) == "c'"
    assert midi_to_abc_pitch(61) == "^C"
    assert midi_to_abc_pitch(69) == "A"


def test_midi_to_abc_pitch_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        midi_to_abc_pitch(128)
    with pytest.raises(ValueError):
        midi_to_abc_pitch(-1)


def test_session_state_to_abc_headers_include_key_meter_tempo_and_note_length() -> None:
    abc = session_state_to_abc(_session_state())

    assert abc.startswith("X:1\n")
    assert "T:HumMuse Session" in abc
    assert "M:4/4" in abc
    assert "L:1/4" in abc
    assert "Q:1/4=110" in abc
    assert "K:Cmaj" in abc


def test_session_state_to_abc_emits_chord_annotations_and_bar_lines() -> None:
    notes = [
        _note(60, onset=0.0, duration=2.0),
        _note(64, onset=2.0, duration=2.0),
        _note(67, onset=4.0, duration=4.0),
    ]
    abc = session_state_to_abc(_session_state(notes=notes, chords=["C", "G"]))

    body = abc.split("K:Cmaj\n", 1)[1]
    bars = [line for line in body.splitlines() if line.strip().endswith("|")]
    assert len(bars) == 2
    assert bars[0].startswith('"C"')
    assert bars[1].startswith('"G"')
    assert "C2" in bars[0]
    assert "E2" in bars[0]
    assert "G4" in bars[1]


def test_session_state_to_abc_pads_partial_bars_with_rest() -> None:
    notes = [_note(60, onset=0.0, duration=1.0)]
    abc = session_state_to_abc(_session_state(notes=notes, chords=["C"]))

    body_line = abc.strip().splitlines()[-1]
    assert body_line.startswith('"C"')
    assert "z" in body_line


def test_session_state_to_abc_with_empty_session_returns_comment_body() -> None:
    abc = session_state_to_abc(_session_state())

    assert "% (empty session" in abc


def test_session_state_to_abc_respects_max_bars_option() -> None:
    notes = [_note(60, onset=float(i) * 4.0, duration=1.0) for i in range(8)]
    abc = session_state_to_abc(
        _session_state(notes=notes, chords=["C"] * 8),
        options=ABCSerializationOptions(max_bars=3),
    )

    bars = [line for line in abc.splitlines() if line.strip().endswith("|")]
    assert len(bars) == 3


def test_session_state_to_abc_falls_back_to_defaults_when_metadata_missing() -> None:
    state = SessionState(session_id=uuid4(), melody_notes=[_note(60, 0.0, 1.0)])

    abc = session_state_to_abc(state)

    assert "Q:1/4=120" in abc
    assert "K:Cmaj" in abc
