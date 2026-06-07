"""Refinement parsing prompt template."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ml.gpt.context_builder import SessionContext
from pydantic import ValidationError
from shared.schemas import RefinementPlan

REFINEMENT_TARGETS = ("harmonizer", "melody_generator", "lyric_generator", "session")
LOGGER = logging.getLogger(__name__)

_HARMONIZER_PARAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requested_target": {"type": "string", "minLength": 1},
        "chord_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "extensions": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "emotion_valence": {"type": "number", "minimum": -1, "maximum": 1},
        "emotion_valence_delta": {"type": "number", "minimum": -1, "maximum": 1},
        "prefer_minor_color": {"type": "boolean"},
        "reduce_minor_bias": {"type": "boolean"},
        "section": {"type": "string", "minLength": 1},
        "tension": {"type": "string", "enum": ["lower", "low", "medium", "higher", "high"]},
    },
}

_MELODY_PARAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requested_target": {"type": "string", "minLength": 1},
        "contour": {"type": "string", "enum": ["rising", "falling", "arch", "valley", "varied"]},
        "rhythmic_density": {"type": "string", "enum": ["lighter", "steady", "denser"]},
        "length": {"type": "string", "enum": ["shorter", "longer", "unchanged"]},
        "section": {"type": "string", "minLength": 1},
        "smooth_contour": {"type": "boolean"},
        "register_shift": {"type": "string", "enum": ["slightly_up", "slightly_down", "up", "down", "none"]},
    },
}

_LYRIC_PARAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requested_target": {"type": "string", "minLength": 1},
        "imagery": {"type": "string", "minLength": 1},
        "num_options": {"type": "integer", "minimum": 1, "maximum": 5},
        "preserve_syllable_targets": {"type": "boolean"},
        "section": {"type": "string", "minLength": 1},
        "language": {"type": "string", "enum": ["en", "ru"]},
        "length": {"type": "string", "enum": ["shorter", "longer", "unchanged"]},
        "preserve_phrase": {"type": "string", "minLength": 1},
    },
}

_SESSION_PARAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requested_target": {"type": "string", "minLength": 1},
        "preserve": {"type": "array", "items": {"type": "string", "minLength": 1}},
        # `emotion` is the single whole-session mood channel: a free-form label
        # (sibling of `genre`) plus optional numeric valence/arousal. When the
        # user expresses a global mood, emit `emotion` AND
        # `emotion_valence`/`emotion_arousal` so the session emotion vector is
        # driven by your own inference rather than a keyword table — this lets
        # nuanced labels ("wistful", "bittersweet") move the vector. Omit the
        # numbers when no emotional shift is implied.
        "emotion": {"type": "string", "minLength": 1},
        "emotion_valence": {"type": "number", "minimum": -1, "maximum": 1},
        "emotion_arousal": {"type": "number", "minimum": -1, "maximum": 1},
        "genre": {"type": "string", "minLength": 1},
    },
}

REFINEMENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operations", "interpretation"],
    "properties": {
        "operations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "params", "rationale"],
                "properties": {
                    "target": {"type": "string", "enum": list(REFINEMENT_TARGETS)},
                    "params": {"type": "object"},
                    "rationale": {"type": "string", "minLength": 1},
                },
                "allOf": [
                    {
                        "if": {"properties": {"target": {"const": "harmonizer"}}},
                        "then": {"properties": {"params": _HARMONIZER_PARAM_SCHEMA}},
                    },
                    {
                        "if": {"properties": {"target": {"const": "melody_generator"}}},
                        "then": {"properties": {"params": _MELODY_PARAM_SCHEMA}},
                    },
                    {
                        "if": {"properties": {"target": {"const": "lyric_generator"}}},
                        "then": {"properties": {"params": _LYRIC_PARAM_SCHEMA}},
                    },
                    {
                        "if": {"properties": {"target": {"const": "session"}}},
                        "then": {"properties": {"params": _SESSION_PARAM_SCHEMA}},
                    },
                ],
            },
        },
        "interpretation": {"type": "string", "minLength": 1},
    },
}
REFINEMENT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "refinement_plan",
        "strict": True,
        "schema": REFINEMENT_RESPONSE_SCHEMA,
    },
}

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "jazzier_chords",
        "instruction": "make it jazzier",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"chord_complexity": "high", "extensions": ["7ths", "9ths"]},
                    "rationale": "Jazzier primarily changes harmonic vocabulary.",
                }
            ],
            "interpretation": "Increase harmonic color while preserving the current song material.",
        },
    },
    {
        "name": "darker_harmony",
        "instruction": "make the chorus darker",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"emotion_valence": -0.35, "prefer_minor_color": True},
                    "rationale": "Darker implies lower valence and more minor/modal harmony.",
                }
            ],
            "interpretation": "Reharmonize the chorus toward a darker emotional color.",
        },
    },
    {
        "name": "stronger_hook",
        "instruction": "give the hook more lift",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "melody_generator",
                    "params": {"contour": "rising", "register_shift": "slightly_up"},
                    "rationale": "More lift is a melody-contour request.",
                }
            ],
            "interpretation": "Make the hook rise more clearly.",
        },
    },
    {
        "name": "two_lyrics",
        "instruction": "write 2 lyric options for this",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "lyric_generator",
                    "params": {"num_options": 2, "preserve_syllable_targets": True},
                    "rationale": "The user asked for lyric alternatives.",
                }
            ],
            "interpretation": "Generate two lyric options that fit the current context.",
        },
    },
    {
        "name": "multi_step_sadder_shorter_lyrics",
        "instruction": "make chorus sadder and shorter, then write 2 lyric options",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"emotion_valence": -0.45, "section": "chorus"},
                    "rationale": "Sadder is first a harmony/emotion adjustment.",
                },
                {
                    "target": "melody_generator",
                    "params": {"section": "chorus", "length": "shorter"},
                    "rationale": "Shorter changes phrase length before generating lyrics.",
                },
                {
                    "target": "lyric_generator",
                    "params": {"num_options": 2, "section": "chorus"},
                    "rationale": "After the musical edits, produce two lyric options.",
                },
            ],
            "interpretation": "Apply sadder chorus harmony, shorten the chorus phrase, then draft two lyric options.",
        },
    },
    {
        "name": "multi_step_reharmonize_then_melody",
        "instruction": "first make the bridge tense, then smooth the melody",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"section": "bridge", "tension": "higher"},
                    "rationale": "The first requested step is harmonic tension.",
                },
                {
                    "target": "melody_generator",
                    "params": {"section": "bridge", "smooth_contour": True},
                    "rationale": "The second requested step smooths the melody.",
                },
            ],
            "interpretation": "Make the bridge harmonically tense, then smooth its melodic contour.",
        },
    },
    {
        "name": "multi_turn_less_sad",
        "instruction": "make it less sad",
        "chat_history": [
            {"role": "user", "content": "Make the verse sadder."},
            {"role": "assistant", "content": "I lowered valence and preferred minor harmony."},
        ],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"emotion_valence_delta": 0.25, "reduce_minor_bias": True},
                    "rationale": "Less sad is relative to the prior sadder-harmony turn.",
                }
            ],
            "interpretation": "Partially undo the previous sadder harmony shift.",
        },
    },
    {
        "name": "multi_turn_shorter_than_previous",
        "instruction": "shorter than that, and keep the catchy line",
        "chat_history": [
            {"role": "user", "content": "Write a catchy chorus option."},
            {"role": "assistant", "content": "I suggested: We burn bright under midnight."},
        ],
        "output": {
            "operations": [
                {
                    "target": "lyric_generator",
                    "params": {"length": "shorter", "preserve_phrase": "catchy line"},
                    "rationale": "The request compares against the prior lyric option in chat history.",
                }
            ],
            "interpretation": "Shorten the previous lyric idea while retaining its hook quality.",
        },
    },
    {
        "name": "whole_session_reset",
        "instruction": "keep the melody but reset the mood to hopeful",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "session",
                    "params": {
                        "preserve": ["melody"],
                        "emotion": "hopeful",
                        "emotion_valence": 0.45,
                        "emotion_arousal": 0.35,
                    },
                    "rationale": "This changes global session intent while preserving melody.",
                }
            ],
            "interpretation": "Update the session mood while keeping the melody fixed.",
        },
    },
    {
        "name": "nuanced_session_mood",
        "instruction": "give the whole thing a wistful, bittersweet feel",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "session",
                    "params": {
                        "emotion": "wistful",
                        "emotion_valence": -0.2,
                        "emotion_arousal": -0.25,
                    },
                    "rationale": "Wistful/bittersweet is a slightly negative, low-energy global mood.",
                }
            ],
            "interpretation": "Set a wistful, bittersweet emotional tone across the session.",
        },
    },
    {
        "name": "russian_multi_step_sadder_shorter_lyrics",
        "instruction": "сделай припев грустнее и короче, потом напиши 2 варианта текста",
        "chat_history": [],
        "output": {
            "operations": [
                {
                    "target": "harmonizer",
                    "params": {"emotion_valence": -0.45, "section": "chorus"},
                    "rationale": "Russian 'грустнее' maps to a sadder harmony/emotion adjustment.",
                },
                {
                    "target": "melody_generator",
                    "params": {"section": "chorus", "length": "shorter"},
                    "rationale": "Russian 'короче' requests a shorter phrase.",
                },
                {
                    "target": "lyric_generator",
                    "params": {"num_options": 2, "section": "chorus", "language": "ru"},
                    "rationale": "Russian 'напиши 2 варианта текста' asks for two lyric options.",
                },
            ],
            "interpretation": "Сделать припев грустнее, укоротить его, затем написать два варианта текста.",
        },
    },
]

_SYSTEM_PROMPT = """You are HumMuse's refinement parser.
Convert the user's natural-language request into an ordered RefinementPlan.
Return only valid JSON matching the schema: {"operations": [{"target": enum, "params": object, "rationale": string}], "interpretation": string}.
Allowed operation targets: harmonizer, melody_generator, lyric_generator, session.
Accept user instructions in English or Russian. Preserve the user's language in interpretation when practical.
Use chat_history to resolve relative instructions such as "less sad", "shorter than that", or "keep it like before".
If the user asks for multiple actions, preserve the requested order.
"""


@dataclass(frozen=True)
class RefinementPrompt:
    messages: list[dict[str, str]]
    response_schema: dict[str, Any]
    response_format: dict[str, Any]


class RefinementPlanValidationError(ValueError):
    """Structured refinement-plan validation failure."""

    def __init__(self, errors: list[dict[str, Any]], *, raw_plan: Any) -> None:
        self.errors = errors
        self.raw_plan = raw_plan
        super().__init__("RefinementPlan validation failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "refinement_plan_validation_failed",
            "errors": self.errors,
            "correction_message": build_refinement_correction_message(self),
        }


def build_refinement_prompt(
    context: SessionContext,
    *,
    instruction: str,
    target_hint: str | None = None,
) -> RefinementPrompt:
    """Build an OpenAI-compatible prompt for parsing refinement instructions."""
    if context.use_case != "refine":
        raise ValueError("Refinement prompts require a refine SessionContext")
    if not instruction.strip():
        raise ValueError("instruction must not be empty")

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(_example_input(example), ensure_ascii=False, sort_keys=True)})
        messages.append({"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False, sort_keys=True)})

    payload = {
        "task": "parse_refinement",
        "instruction": instruction.strip(),
        "target_hint": target_hint,
        "required_output": REFINEMENT_RESPONSE_SCHEMA,
        "context": {
            "melody_profile": context.fields.get("melody_profile"),
            "chord_progressions": context.fields.get("chord_progressions"),
            "syllable_targets_per_line": context.fields.get("syllable_targets_per_line"),
            "previous_lyrics": context.fields.get("previous_lyrics"),
            "emotion_vector": context.fields.get("emotion_vector"),
            "chat_history": context.fields.get("recent_chat_history"),
            "last_refinement_plan_results": context.fields.get("last_refinement_plan_results"),
        },
    }
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)})
    return RefinementPrompt(
        messages=messages,
        response_schema=REFINEMENT_RESPONSE_SCHEMA,
        response_format=REFINEMENT_RESPONSE_FORMAT,
    )


def parse_refinement_plan(raw_plan: Any) -> RefinementPlan:
    """Parse and strictly validate an LLM refinement-plan response."""
    data = _coerce_plan_payload(raw_plan)
    try:
        return RefinementPlan.model_validate(data)
    except ValidationError as exc:
        raise RefinementPlanValidationError(_clean_validation_errors(exc.errors(include_url=False)), raw_plan=data) from exc


def generate_refinement_plan_with_retries(
    generator,
    prompt: RefinementPrompt,
    *,
    max_retries: int = 2,
    logger: Any | None = LOGGER,
) -> RefinementPlan:
    """Generate, validate, and retry malformed refinement plans with logged correction prompts."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    messages = [dict(message) for message in prompt.messages]
    last_error: RefinementPlanValidationError | None = None
    for attempt in range(max_retries + 1):
        raw_response = generator(messages, response_format=prompt.response_format)
        try:
            return parse_refinement_plan(_extract_response_content(raw_response))
        except RefinementPlanValidationError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            correction_message = build_refinement_correction_message(exc)
            _log_retry_with_correction(logger, attempt=attempt + 1, error=exc, correction_message=correction_message)
            messages.append({"role": "user", "content": correction_message})
    raise last_error or RuntimeError("refinement generation failed without a validation error")


def build_refinement_correction_message(error: RefinementPlanValidationError) -> str:
    return (
        "Your previous refinement plan did not match the strict schema. "
        "Return corrected JSON only. Validation errors: "
        f"{json.dumps(error.errors, ensure_ascii=False, sort_keys=True)}"
    )


def _coerce_plan_payload(raw_plan: Any) -> Any:
    if hasattr(raw_plan, "content"):
        raw_plan = raw_plan.content
    if isinstance(raw_plan, str):
        try:
            return json.loads(raw_plan)
        except json.JSONDecodeError as exc:
            raise RefinementPlanValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ["body"],
                        "msg": str(exc),
                        "input": raw_plan,
                    }
                ],
                raw_plan=raw_plan,
            ) from exc
    return raw_plan


def _extract_response_content(raw_response: Any) -> Any:
    if hasattr(raw_response, "content"):
        return raw_response.content
    return raw_response


def _clean_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_errors = []
    for error in errors:
        clean = {}
        for key, value in error.items():
            if key == "ctx" and isinstance(value, dict):
                clean[key] = {ctx_key: str(ctx_value) for ctx_key, ctx_value in value.items()}
            else:
                clean[key] = value
        clean_errors.append(clean)
    return clean_errors


def _log_retry_with_correction(
    logger: Any | None,
    *,
    attempt: int,
    error: RefinementPlanValidationError,
    correction_message: str,
) -> None:
    if logger is None:
        return
    try:
        logger.info(
            "refinement_plan_retry_with_correction",
            attempt=attempt,
            errors=error.errors,
            correction_message=correction_message,
        )
    except TypeError:
        logger.info(
            "refinement_plan_retry_with_correction",
            extra={
                "attempt": attempt,
                "errors": error.errors,
                "correction_message": correction_message,
            },
        )


def _example_input(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "parse_refinement",
        "instruction": example["instruction"],
        "context": {
            "chat_history": example["chat_history"],
            "melody_profile": {"contour": "arch", "rhythmic_density": 1.2},
            "chord_progressions": [{"chords": ["C", "Am", "F", "G"], "roman_numerals": ["I", "vi", "IV", "V"]}],
            "syllable_targets_per_line": [4, 5],
            "previous_lyrics": "Hold the line\nUnder city lights",
            "emotion_vector": {"valence": 0.1, "arousal": 0.4},
            "last_refinement_plan_results": None,
        },
    }
