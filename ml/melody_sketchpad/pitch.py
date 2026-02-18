"""Pitch extraction stub for melody sketchpad."""

from __future__ import annotations


def seed_from_size(size: int) -> int:
    return max(1, min(size // 1000, 4))

