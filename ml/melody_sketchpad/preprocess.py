"""Preprocessing helpers for hummed audio input."""

from __future__ import annotations


def audio_size(audio_bytes: bytes) -> int:
    """Return deterministic size feature used by mock pipeline."""
    return len(audio_bytes)

