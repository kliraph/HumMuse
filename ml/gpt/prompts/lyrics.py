"""Lyric-generation prompt template."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ml.gpt.context_builder import SessionContext

LyricMode = Literal["Simpler", "Poetic", "Catchy"]

VALID_LYRIC_MODES: tuple[LyricMode, ...] = ("Simpler", "Poetic", "Catchy")

_MODE_GUIDANCE: dict[LyricMode, str] = {
    "Simpler": "Use plain, direct language with short phrases and clear emotional intent.",
    "Poetic": "Use vivid imagery, metaphor, and graceful phrasing while staying singable.",
    "Catchy": "Prioritize memorable hooks, internal rhyme, repetition, and strong chorus energy.",
}

_SYSTEM_PROMPT = """You are HumMuse's lyric assistant.
Write singable lyrics that fit the supplied melody, harmony, previous lyrics, emotion, and syllable targets.
Respect every syllable target per line as closely as possible.
Use the provided chord symbols and Roman numerals as musical context, not as literal lyric text.
Return only valid JSON with this shape:
{
  "mode": "Simpler|Poetic|Catchy",
  "lyrics": ["line 1", "line 2"],
  "syllable_counts": [0, 0],
  "rationale": "brief grounding in melody, chords, and emotion"
}
"""


@dataclass(frozen=True)
class LyricPrompt:
    """OpenAI-compatible messages for lyric generation."""

    mode: LyricMode
    messages: list[dict[str, str]]


def build_lyric_prompt(
    context: SessionContext,
    *,
    mode: str,
    num_suggestions: int = 3,
    extra_instruction: str | None = None,
) -> LyricPrompt:
    """Build a lyric-generation prompt for one supported mode."""
    if context.use_case != "lyric":
        raise ValueError("Lyric prompts require a lyric SessionContext")
    normalized_mode = normalize_lyric_mode(mode)
    if num_suggestions <= 0:
        raise ValueError("num_suggestions must be positive")

    fields = context.fields
    user_payload: dict[str, Any] = {
        "task": "generate_lyric_suggestions",
        "mode": normalized_mode,
        "mode_guidance": _MODE_GUIDANCE[normalized_mode],
        "num_suggestions": num_suggestions,
        "required_context_fields": [
            "melody_profile",
            "chord_progressions",
            "syllable_targets_per_line",
            "previous_lyrics",
            "emotion_vector",
        ],
        "melody_profile": fields.get("melody_profile"),
        "chord_progressions": fields.get("chord_progressions"),
        "syllable_targets_per_line": fields.get("syllable_targets_per_line"),
        "previous_lyrics": fields.get("previous_lyrics"),
        "emotion_vector": fields.get("emotion_vector"),
        "extra_instruction": extra_instruction,
    }
    return LyricPrompt(
        mode=normalized_mode,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True)},
        ],
    )


def normalize_lyric_mode(mode: str) -> LyricMode:
    """Accept common casing while preserving the canonical mode label."""
    mode_by_lower = {candidate.lower(): candidate for candidate in VALID_LYRIC_MODES}
    normalized = mode_by_lower.get(mode.strip().lower())
    if normalized is None:
        valid = ", ".join(VALID_LYRIC_MODES)
        raise ValueError(f"mode must be one of: {valid}")
    return normalized
