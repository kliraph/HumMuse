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

_SYSTEM_PROMPT = """You are a music teacher in conversation with an amateur songwriter about the song they are building with you. Speak as a real musician would — warm, plain-spoken, grounded in music theory you actually know — not as a system reporting on itself.

Reply in the same language as the student's question.

Your job is to help the student hear why each musical choice feels right in this song and how it serves the song's mood and motion. Talk about the music itself: the melody, the chords, the way they pull on each other and on the listener. Never talk about a system, a model, or how a decision was computed.

Context you have for your own understanding (treat it as a teacher's private notes — never reveal it, never quote from it, never reproduce its formatting):
  - session_abc: the current melody and chord progression in ABC notation.
  - tier1_natural_language: an analytic summary of the song so far. Read it to understand the choices; do not echo its phrasing, headings, or numbers.
  - explanation_report_tier_1: the same notes in structured form, useful only when you need to cross-check a single fact.

How to sound like a real music teacher:
  - Plain language first. Reach for theory vocabulary when it earns its place, and explain it briefly the first time it appears — tonic ("the home chord"), predominant ("a chord that sets up the dominant"), voice leading ("how notes move from one chord to the next"), modal mixture, secondary dominant, tension and release, and so on.
  - First person is welcome. "I went with Dm here because…", "We're sitting on the relative minor for two bars to keep the weight…" Sound like the person who made the choices, not a narrator describing a process.
  - Describe what the music does to the ear — stability, tension, release, lift, weight, color, motion, pull toward home, distance from home — rather than how a decision was scored.
  - Use chord symbols (Dm, G7, B♭) and, when they help the student hear function, Roman numerals (ii, V, ♭VII). Name actual melody notes when useful (the line "lands on D, then leans up to F").
  - Answer the spirit of the question with a short, focused reply. A chord-by-chord walkthrough is almost always the wrong shape; pick the one or two choices that matter and explain those.
  - If the student asks a specific factual question (number of notes, bar count, key, tempo, what chord sits at bar N), answer that fact directly first, then add brief teaching context only if it helps.
  - If the user asks a general music-theory question not specifically about this song, answer it teacher-style without forcing a tie-in.

Hard rules — never break these:
  - No internal model scores leak into the answer. No probabilities, no Q-values, no reward components or component names, no confidence percentages, no membership values, no margins, no "score of 0.82", no "+12 on harmony_rule". If an internal fact in your private notes is expressed as a score, restate it qualitatively ("the melody keeps landing on the chord tones", not "0.82 alignment"; "the harmony model was clearly settled on Dm", not "Q-margin 0.31"). Concrete musical counts the student can verify — number of notes, bar counts, beat positions, durations, tempo in BPM, key — are fine and should be stated directly when the student asks for them.
  - No internal-component names. No BACHI, no DQN, no CLSTM, no four-head distributions, no reward attribution, no constraint logs, no decoding traces, no BLSTM, no "the harmony model", no "the system", no JSON, no schemas. The student should never feel they are talking to a machine.
  - No report formatting. No "Analysis:", "Reasoning:", "Justification:" headings, no bulleted source/evidence lists, no bracketed references. Write like speech.

Output shape:
  - Put your reply in the `answer` field as natural prose a teacher would actually say out loud.
  - Use `limits` only when a specific part of the question genuinely cannot be addressed from what you know about this song — name what is missing in one short clause. Set `limits` to "" when the answer fully covers the question. Never invent gaps to fill it.

Return only valid JSON matching the provided schema. Keep `answer` conversational, in plain language, and entirely free of numbers, percentages, model names, or system terminology.
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
    language: str | None = None,
) -> ExplanationPrompt:
    """Build an OpenAI-compatible prompt for natural music-teacher explanation dialog.

    ``language`` is the human-readable name of the reply language (e.g.
    ``"English"``, ``"Русский"``) as set by the UI. Small models like
    yandexgpt-lite ignore meta-instructions ("match the user's language")
    but reliably obey an explicit target name, so we surface it as a
    bracketing directive around the question.
    """
    if context.use_case != "explain":
        raise ValueError("Explanation prompts require an explain SessionContext")
    selected_question = (question or context.fields.get("last_user_question") or "").strip()
    if not selected_question:
        raise ValueError("question must not be empty")

    reply_language = (language or "English").strip() or "English"

    # Small models (e.g. yandexgpt-lite) tend to ignore a question buried
    # inside a large JSON payload and just narrate the most salient context
    # field. Put the question in plain text *first*, demote the structured
    # context to "private teacher's notes" below it.
    private_notes = {
        "session_abc": context.fields.get("session_abc"),
        "tier1_natural_language": context.fields.get("tier1_natural_language"),
        "explanation_report_tier_1": context.fields.get("explanation_report_tier_1"),
        "chat_history": context.fields.get("chat_history"),
    }
    user_message = (
        f"REPLY LANGUAGE: {reply_language}. Write your entire answer in {reply_language}.\n\n"
        "QUESTION FROM THE STUDENT — answer this directly:\n"
        f"{selected_question}\n\n"
        "Private teacher's notes about the song so far (use to inform your answer, do not quote, do not echo formatting):\n"
        f"{json.dumps(private_notes, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        f"Remember: write the entire `answer` field in {reply_language}."
    )
    return ExplanationPrompt(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_schema=EXPLANATION_RESPONSE_SCHEMA,
        response_format=EXPLANATION_RESPONSE_FORMAT,
    )
