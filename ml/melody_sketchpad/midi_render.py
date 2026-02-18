"""MIDI rendering helpers for sketchpad output."""

from __future__ import annotations


def render_midi_stub(audio_size: int) -> bytes:
    return b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60" + bytes([audio_size % 128])

