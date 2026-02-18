"""Mapping mood to mock chord progressions."""

from __future__ import annotations


def top_progressions_for_mood(mood: str) -> list[list[str]]:
    if mood == "hopeful":
        return [
            ["C", "G", "Am", "F"],
            ["Am", "F", "C", "G"],
            ["Dm", "G", "C", "Am"],
        ]
    return [
        ["Em", "C", "G", "D"],
        ["Am", "Em", "F", "C"],
        ["Dm", "Am", "Bb", "F"],
    ]

