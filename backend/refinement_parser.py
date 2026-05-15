"""Refinement instruction parsing: GPT first, keyword heuristic fallback.

Wraps :class:`ml.gpt.pipeline.GPTPipeline` for the ``refine`` use case and
degrades to :func:`backend.mock_pipeline.build_refinement_plan` whenever the
GPT path is unavailable (missing config, transport error) or produces a
payload that fails strict schema validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from backend.logging_config import get_logger
from backend.mock_pipeline import build_refinement_plan as _fallback_build_refinement_plan
from ml.gpt.pipeline import GPTPipeline
from ml.gpt.prompts.refine import RefinementPlanValidationError
from pydantic import ValidationError
from shared.schemas import RefinementPlan, SessionState

LOGGER = get_logger("backend.refinement_parser")

ParseSource = Literal["gpt", "fallback"]


@dataclass(frozen=True)
class ParsedRefinement:
    """Result of parsing a natural-language refinement instruction."""

    plan: RefinementPlan
    source: ParseSource
    model_version: str | None
    cache_hit: bool
    latency_ms: float | None
    error: str | None


def parse_refinement_instruction(
    state: SessionState,
    instruction: str,
    *,
    target_hint: str | None = None,
    pipeline: GPTPipeline | None = None,
) -> ParsedRefinement:
    """Parse a refinement instruction, returning a structured plan and metadata.

    The GPT pipeline is attempted first. On any failure — pipeline construction
    error, GPT transport error, or strict-schema validation failure — the
    keyword heuristic is used so callers always receive a usable plan.
    """
    if pipeline is None:
        pipeline = _safe_build_pipeline(instruction, target_hint)
    if pipeline is None:
        return _fallback(instruction, target_hint, reason="pipeline_unavailable")

    try:
        result = pipeline.route_request(
            "refine",
            state,
            instruction,
            target_hint=target_hint,
        )
    except RefinementPlanValidationError as exc:
        LOGGER.info(
            "refinement_parse_validation_failed",
            errors=exc.errors,
            instruction=instruction,
            target_hint=target_hint,
        )
        return _fallback(instruction, target_hint, reason="validation_failed")
    except Exception as exc:  # GPT transport / SDK / auth failures all degrade
        LOGGER.info(
            "refinement_parse_gpt_failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
            instruction=instruction,
            target_hint=target_hint,
        )
        return _fallback(instruction, target_hint, reason=f"transport:{exc.__class__.__name__}")

    plan = _coerce_plan(result.parsed)
    if plan is None:
        LOGGER.info(
            "refinement_parse_unexpected_payload",
            payload_type=type(result.parsed).__name__,
            instruction=instruction,
            target_hint=target_hint,
        )
        return _fallback(instruction, target_hint, reason="unexpected_payload")

    LOGGER.info(
        "refinement_parse_gpt_succeeded",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        operation_count=len(plan.operations),
    )
    return ParsedRefinement(
        plan=plan,
        source="gpt",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        error=None,
    )


def _safe_build_pipeline(instruction: str, target_hint: str | None) -> GPTPipeline | None:
    try:
        return GPTPipeline.from_env()
    except Exception as exc:
        LOGGER.info(
            "refinement_parse_pipeline_unavailable",
            error=str(exc),
            error_type=exc.__class__.__name__,
            instruction=instruction,
            target_hint=target_hint,
        )
        return None


def _coerce_plan(parsed: Any) -> RefinementPlan | None:
    """Accept either a RefinementPlan instance or a dict (e.g. from cache)."""
    if isinstance(parsed, RefinementPlan):
        return parsed
    if isinstance(parsed, dict):
        try:
            return RefinementPlan.model_validate(parsed)
        except ValidationError:
            return None
    return None


def _fallback(instruction: str, target_hint: str | None, *, reason: str) -> ParsedRefinement:
    plan = _fallback_build_refinement_plan(instruction, target_hint)
    return ParsedRefinement(
        plan=plan,
        source="fallback",
        model_version=None,
        cache_hit=False,
        latency_ms=None,
        error=reason,
    )
