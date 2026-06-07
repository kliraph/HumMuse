"""Mood-inference prompt template.

Given a block of lyrics, ask the model for both:

* a continuous emotion vector — ``valence`` and ``arousal`` in ``[-1, 1]`` —
  that the DQN harmonizer can consume directly, and
* a free-form ``mood_label`` string in the same language as the lyrics, so
  the UI can surface a richer description than the 6-class taxonomies you
  get from a small classifier.

Yandex strict-mode requires every property to be in ``required``; the
schema below mirrors that constraint. The model is told to emit
``rationale`` even when it has nothing notable to say (single phrase is
fine), keeping the response a single shape regardless of input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ml.gpt.context_builder import SessionContext


MOOD_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["valence", "arousal", "mood_label", "rationale"],
    "properties": {
        "valence": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Pleasantness on a continuous scale; -1 is the saddest, +1 is the most joyful.",
        },
        "arousal": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Activation/energy on a continuous scale; -1 is the calmest, +1 is the most intense.",
        },
        "mood_label": {
            "type": "string",
            "minLength": 1,
            "description": (
                "A free-form mood label (1-4 words) in the same language as the lyrics. "
                "Examples: 'wistful', 'triumphant', 'печальный с проблесками надежды'. "
                "Avoid 6-class taxonomies — pick the phrase that best fits."
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "One short sentence (or empty string when nothing notable applies) "
                "grounding the mood in concrete lyric details."
            ),
        },
    },
}

MOOD_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mood_inference",
        "strict": True,
        "schema": MOOD_RESPONSE_SCHEMA,
    },
}


_SYSTEM_PROMPT = """You are HumMuse's mood-inference module.
Given a block of lyrics, return a continuous emotion vector plus a free-form mood label.
Rules:
  - Read the lyrics holistically. Mixed moods (bittersweet, anxious-hopeful) are common and should be reflected in the vector — do not snap to a stereotypical label.
  - `valence` runs from -1 (sad, dark, despairing) through 0 (neutral) to +1 (joyful, bright, triumphant).
  - `arousal` runs from -1 (calm, still, contemplative) through 0 (steady) to +1 (energetic, intense, urgent).
  - `mood_label` is a short free-form phrase (1-4 words) in the SAME language as the lyrics. Match the lyrics' language even if it differs from this system prompt's language. Examples: "wistful", "yearning", "triumphant", "печальный с проблесками надежды", "тоскующий", "взрывной".
  - `rationale` is one short sentence that grounds your reading in concrete lyric details. Use "" when there is nothing notable to say.
  - Return ONLY valid JSON matching the response schema. No prose outside the JSON.
"""


@dataclass(frozen=True)
class MoodPrompt:
    messages: list[dict[str, str]]
    response_schema: dict[str, Any]
    response_format: dict[str, Any]


def build_mood_prompt(context: SessionContext, *, language: str | None = None) -> MoodPrompt:
    """Build an OpenAI-compatible prompt for emotion-vector + mood-label inference."""
    if context.use_case != "mood":
        raise ValueError("Mood prompts require a mood SessionContext")

    lyrics_text = (context.fields.get("lyrics_text") or "").strip()
    if not lyrics_text:
        raise ValueError("lyrics_text must not be empty")

    payload: dict[str, Any] = {
        "task": "infer_mood_from_lyrics",
        "lyrics": lyrics_text,
        "expected_output": MOOD_RESPONSE_SCHEMA,
    }
    if language:
        payload["language_hint"] = language

    return MoodPrompt(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)},
        ],
        response_schema=MOOD_RESPONSE_SCHEMA,
        response_format=MOOD_RESPONSE_FORMAT,
    )
