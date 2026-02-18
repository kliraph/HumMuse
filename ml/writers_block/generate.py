"""Text generation helpers for writer's block module."""

from __future__ import annotations


def build_suggestions(target_mood: str) -> list[str]:
    return [
        f"Continue with an image-driven line that matches a {target_mood} tone.",
        "Rewrite the last line with a stronger verb and fewer adjectives.",
        "Add a contrasting second line to create lyrical tension before the chorus.",
    ]

