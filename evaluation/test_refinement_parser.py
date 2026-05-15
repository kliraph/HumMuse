"""Tests for backend.refinement_parser: GPT-first parsing with keyword fallback."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from backend.refinement_parser import ParsedRefinement, parse_refinement_instruction
from ml.gpt.pipeline import GPTPipelineResult
from ml.gpt.prompts.refine import RefinementPlanValidationError
from shared.schemas import RefinementOp, RefinementPlan, SessionState


def _session() -> SessionState:
    return SessionState(session_id=uuid4())


def _gpt_plan() -> RefinementPlan:
    return RefinementPlan(
        operations=[
            RefinementOp(
                target="harmonizer",
                params={"emotion_valence": -0.4, "prefer_minor_color": True},
                rationale="Darker harmony shift.",
            )
        ],
        interpretation="Reharmonize toward a darker emotional color.",
    )


class _FakePipeline:
    def __init__(self, *, result: GPTPipelineResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def route_request(self, use_case: str, state: SessionState, instruction: str, **kwargs: Any) -> GPTPipelineResult:
        self.calls.append({"use_case": use_case, "instruction": instruction, **kwargs})
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


def _gpt_result(parsed: Any, *, cache_hit: bool = False) -> GPTPipelineResult:
    return GPTPipelineResult(
        use_case="refine",
        content="{}",
        parsed=parsed,
        cache_hit=cache_hit,
        provider="yandex",
        model_version="test-model",
        latency_ms=12.5,
        prompt_token_count=100,
        completion_token_count=20,
        eval_mode=False,
    )


def test_gpt_success_returns_plan_with_metadata() -> None:
    plan = _gpt_plan()
    pipeline = _FakePipeline(result=_gpt_result(plan))

    parsed = parse_refinement_instruction(
        _session(),
        "make it darker",
        target_hint="chords",
        pipeline=pipeline,
    )

    assert parsed.source == "gpt"
    assert parsed.plan is plan
    assert parsed.model_version == "test-model"
    assert parsed.cache_hit is False
    assert parsed.latency_ms == 12.5
    assert parsed.error is None
    assert pipeline.calls[0]["use_case"] == "refine"
    assert pipeline.calls[0]["target_hint"] == "chords"


def test_gpt_cache_hit_dict_payload_is_coerced_to_refinement_plan() -> None:
    plan = _gpt_plan()
    pipeline = _FakePipeline(result=_gpt_result(plan.model_dump(), cache_hit=True))

    parsed = parse_refinement_instruction(
        _session(),
        "make it jazzier",
        pipeline=pipeline,
    )

    assert parsed.source == "gpt"
    assert parsed.cache_hit is True
    assert parsed.plan.operations[0].target == "harmonizer"
    assert parsed.plan.interpretation == plan.interpretation


def test_validation_error_falls_back_to_keyword_heuristic() -> None:
    pipeline = _FakePipeline(
        exc=RefinementPlanValidationError([{"type": "missing", "loc": ["operations"]}], raw_plan={}),
    )

    parsed = parse_refinement_instruction(
        _session(),
        "make it darker",
        target_hint="chords",
        pipeline=pipeline,
    )

    assert parsed.source == "fallback"
    assert parsed.error == "validation_failed"
    # Keyword heuristic recognises "dark" → harmonizer op.
    assert any(op.target == "harmonizer" for op in parsed.plan.operations)


def test_transport_error_falls_back_to_keyword_heuristic() -> None:
    pipeline = _FakePipeline(exc=RuntimeError("connection refused"))

    parsed = parse_refinement_instruction(
        _session(),
        "сделай припев грустнее",
        pipeline=pipeline,
    )

    assert parsed.source == "fallback"
    assert parsed.error == "transport:RuntimeError"
    assert any(op.target == "harmonizer" for op in parsed.plan.operations)


def test_unexpected_parsed_payload_falls_back() -> None:
    pipeline = _FakePipeline(result=_gpt_result("not-a-plan"))

    parsed = parse_refinement_instruction(
        _session(),
        "give the hook more lift",
        pipeline=pipeline,
    )

    assert parsed.source == "fallback"
    assert parsed.error == "unexpected_payload"


def test_no_pipeline_available_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no API key configured")

    monkeypatch.setattr("backend.refinement_parser.GPTPipeline.from_env", staticmethod(_raise))

    parsed = parse_refinement_instruction(
        _session(),
        "write 2 lyric options",
    )

    assert parsed.source == "fallback"
    assert parsed.error == "pipeline_unavailable"
    assert any(op.target == "lyric_generator" for op in parsed.plan.operations)


def test_fallback_interpretation_is_non_empty() -> None:
    """The fallback plan must still satisfy the existing API contract."""
    pipeline = _FakePipeline(exc=RuntimeError("boom"))
    parsed = parse_refinement_instruction(_session(), "make it shorter", pipeline=pipeline)
    assert parsed.plan.interpretation.strip()
    assert isinstance(parsed, ParsedRefinement)
