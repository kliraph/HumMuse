"""Text cleaning helpers for lyrics."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()

