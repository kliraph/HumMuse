"""Tests for strict RefinementPlan validation and correction retries."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ml.gpt.prompts.refine import (
    RefinementPlanValidationError,
    build_refinement_prompt,
    generate_refinement_plan_with_retries,
    parse_refinement_plan,
)
from ml.gpt.context_builder import SessionContext
from shared.schemas import RefinementPlan


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.events.append((event, fields))


def _refine_context() -> SessionContext:
    return SessionContext(
        use_case="refine",
        fields={
            "melody_profile": {"contour": "arch"},
            "chord_progressions": [{"chords": ["C", "Am"], "roman_numerals": ["I", "vi"]}],
            "syllable_targets_per_line": [4, 5],
            "previous_lyrics": "Hold the line",
            "emotion_vector": {"valence": 0.0, "arousal": 0.3},
            "recent_chat_history": [],
            "last_refinement_plan_results": None,
        },
    )


def test_refinement_plan_rejects_unknown_target() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RefinementPlan.model_validate(
            {
                "operations": [
                    {
                        "target": "arranger",
                        "params": {},
                        "rationale": "Unknown target.",
                    }
                ],
                "interpretation": "Bad target.",
            }
        )

    assert "arranger" in str(exc_info.value)


def test_refinement_plan_rejects_unknown_param_for_target() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RefinementPlan.model_validate(
            {
                "operations": [
                    {
                        "target": "harmonizer",
                        "params": {"tempo": 300},
                        "rationale": "Tempo is not a harmonizer param.",
                    }
                ],
                "interpretation": "Bad param.",
            }
        )

    assert "Unknown params for harmonizer" in str(exc_info.value)


def test_refinement_plan_rejects_out_of_range_params() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RefinementPlan.model_validate(
            {
                "operations": [
                    {
                        "target": "lyric_generator",
                        "params": {"num_options": 99},
                        "rationale": "Too many options.",
                    }
                ],
                "interpretation": "Bad range.",
            }
        )

    assert "lyric_generator.num_options must be between 1 and 5" in str(exc_info.value)


def test_parse_refinement_plan_raises_structured_error_for_malformed_plan() -> None:
    with pytest.raises(RefinementPlanValidationError) as exc_info:
        parse_refinement_plan(
            {
                "operations": [
                    {
                        "target": "harmonizer",
                        "params": {"emotion_valence": -3},
                        "rationale": "Out of range.",
                    }
                ],
                "interpretation": "Malformed.",
            }
        )

    payload = exc_info.value.to_dict()
    assert payload["error"] == "refinement_plan_validation_failed"
    assert payload["errors"]
    assert "corrected JSON" in payload["correction_message"]


def test_parse_refinement_plan_accepts_json_string() -> None:
    plan = parse_refinement_plan(
        json.dumps(
            {
                "operations": [
                    {
                        "target": "harmonizer",
                        "params": {"emotion_valence": -0.3},
                        "rationale": "Make it darker.",
                    }
                ],
                "interpretation": "Make harmony darker.",
            }
        )
    )

    assert plan.operations[0].target == "harmonizer"


def test_retry_with_correction_is_logged_and_uses_response_format() -> None:
    prompt = build_refinement_prompt(_refine_context(), instruction="make it less sad")
    logger = FakeLogger()
    calls: list[dict] = []
    responses = [
        {"operations": [{"target": "harmonizer", "params": {"emotion_valence": -5}, "rationale": "Bad."}], "interpretation": "Bad."},
        {"operations": [{"target": "harmonizer", "params": {"emotion_valence": 0.2}, "rationale": "Less sad."}], "interpretation": "Less sad."},
    ]

    def generator(messages, *, response_format):
        calls.append({"messages": messages, "response_format": response_format})
        return responses.pop(0)

    plan = generate_refinement_plan_with_retries(generator, prompt, logger=logger)

    assert plan.operations[0].params["emotion_valence"] == 0.2
    assert len(calls) == 2
    assert calls[0]["response_format"] == prompt.response_format
    assert "Validation errors" in calls[1]["messages"][-1]["content"]
    assert logger.events[0][0] == "refinement_plan_retry_with_correction"
    assert logger.events[0][1]["errors"]
