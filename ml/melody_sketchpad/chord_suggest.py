"""Chord suggestion helpers for melody sketchpad."""

from __future__ import annotations

from shared.schemas import ChordSuggestion


def suggest_chords() -> list[ChordSuggestion]:
    return [
        ChordSuggestion(symbol="Am", start_bar=0, beats=4),
        ChordSuggestion(symbol="F", start_bar=1, beats=4),
        ChordSuggestion(symbol="C", start_bar=2, beats=4),
        ChordSuggestion(symbol="G", start_bar=3, beats=4),
    ]

