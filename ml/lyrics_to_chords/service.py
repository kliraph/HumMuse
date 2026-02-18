"""Service entrypoint for lyrics-to-chords mock inference."""

from __future__ import annotations

from ml.lyrics_to_chords.cleaner import normalize_text
from ml.lyrics_to_chords.mapping import top_progressions_for_mood
from ml.lyrics_to_chords.mood import infer_mood
from shared.schemas import ExplanationPart


def generate_from_lyrics(text: str) -> tuple[str, list[list[str]], list[ExplanationPart]]:
    cleaned = normalize_text(text)
    mood = infer_mood(cleaned)
    top_progressions = top_progressions_for_mood(mood)
    explanation = [
        ExplanationPart(
            title="Mood inference",
            detail=f"Keyword-based heuristic classified the lyrics as {mood}.",
        ),
        ExplanationPart(
            title="Cadence",
            detail="Each progression ends with a stable resolution suitable for vocal hooks.",
        ),
    ]
    return mood, top_progressions, explanation

