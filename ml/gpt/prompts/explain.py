"""Explanation-dialog prompt template."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ml.gpt.context_builder import SessionContext

EXPLANATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # Both keys are required so the schema works with providers that enforce
    # strict mode by rejecting optional properties (e.g. Yandex Cloud).
    # `limits` is allowed to be an empty string when nothing meaningful applies.
    "required": ["answer", "limits"],
    "properties": {
        "answer": {
            "type": "string",
            "minLength": 1,
            "description": "Natural-language explanation in a friendly music-teacher voice.",
        },
        "limits": {
            "type": "string",
            "description": (
                "Brief note about anything the supplied session context cannot address. "
                "Set to an empty string when the answer fully covers the question."
            ),
        },
    },
}

EXPLANATION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "teacher_explanation",
        "strict": True,
        "schema": EXPLANATION_RESPONSE_SCHEMA,
    },
}

_SYSTEM_PROMPT = """You are HumMuse's music-teacher conversation partner.
Speak as a warm, plain-spoken music teacher who is helping a curious amateur understand why the system's choices fit the song. Use the supplied session context to understand the choices, then explain their suitability in natural prose — the way you would talk to a student in a lesson, not the way you would write a report.
Available context:
  - session_abc: the current melody and chord progression in ABC notation.
  - tier1_natural_language: prose summary of the underlying analysis, including the CLSTM four-head distributions and DQN reward attributions that drove chord decisions.
  - explanation_report_tier_1: the same data in structured form, available if you want to double-check a detail.
Style rules:
  - Plain language first. Reach for musical-theory terms (tonic, leading tone, predominant, voice leading, tension and release) when they help, but explain what they mean in context.
  - Talk about *why the choice feels right* and how it serves the song, not about probabilities, Q-values, reward components, percentages, or other internal numbers. Use the data to inform your judgement, not to quote it.
  - It is fine — and often better — to skip a chord-by-chord walkthrough and answer the spirit of the question.
  - If the user asks a general music-theory question that is not specifically about this session, answer it teacher-style without forcing a tie-in to the song.
  - The response has two required fields: `answer` and `limits`. Put your reply in `answer`. Use `limits` only when the supplied context genuinely cannot address part of the question — name what is missing in one short clause. Set `limits` to an empty string ("") when the answer fully covers the question. Do not invent details to fill it.
  - Do not mention BACHI, decoding traces, BLSTM distributions, JSON, schemas, or the internal architecture. The shipping chord model is referred to simply as "the harmony model" when needed.
Return only valid JSON matching the provided schema. Keep the answer conversational and free of numeric citations.
"""


@dataclass(frozen=True)
class ExplanationPrompt:
    messages: list[dict[str, str]]
    response_schema: dict[str, Any]
    response_format: dict[str, Any]


def build_explanation_prompt(
    context: SessionContext,
    *,
    question: str | None = None,
) -> ExplanationPrompt:
    """Build an OpenAI-compatible prompt for natural music-teacher explanation dialog."""
    if context.use_case != "explain":
        raise ValueError("Explanation prompts require an explain SessionContext")
    selected_question = (question or context.fields.get("last_user_question") or "").strip()
    if not selected_question:
        raise ValueError("question must not be empty")

    payload = {
        "task": "answer_like_a_music_teacher",
        "question": selected_question,
        "context_provided": [
            "session_abc",
            "tier1_natural_language",
            "explanation_report_tier_1",
            "chat_history",
            "last_user_question",
        ],
        "session_abc": context.fields.get("session_abc"),
        "tier1_natural_language": context.fields.get("tier1_natural_language"),
        "explanation_report_tier_1": context.fields.get("explanation_report_tier_1"),
        "chat_history": context.fields.get("chat_history"),
        "last_user_question": context.fields.get("last_user_question"),
    }
    return ExplanationPrompt(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)},
        ],
        response_schema=EXPLANATION_RESPONSE_SCHEMA,
        response_format=EXPLANATION_RESPONSE_FORMAT,
    )
