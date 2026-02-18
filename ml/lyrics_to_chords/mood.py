"""Mood inference helpers for lyrics."""

from __future__ import annotations


def infer_mood(cleaned_text: str) -> str:
    if "love" in cleaned_text or "sun" in cleaned_text:
        return "hopeful"
    return "reflective"

