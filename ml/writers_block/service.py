"""Service entrypoint for anti-writer's-block mock inference."""

from __future__ import annotations

from ml.writers_block.analyze_lyrics import lyrics_length
from ml.writers_block.analyze_melody import melody_hint
from ml.writers_block.generate import build_suggestions
from shared.schemas import ExplanationPart


def generate_help(text: str, mood: str | None) -> tuple[list[str], list[ExplanationPart]]:
    target_mood = mood or "neutral"
    suggestions = build_suggestions(target_mood)
    explanation = [
        ExplanationPart(
            title="Continuation strategy",
            detail="The suggestions alternate imagery, diction tightening, and contrast.",
        ),
        ExplanationPart(
            title="Constraint",
            detail=f"Prompt context length used: {lyrics_length(text)} characters. {melody_hint()}",
        ),
    ]
    return suggestions, explanation

