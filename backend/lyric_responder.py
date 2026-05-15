"""Generate lyric suggestions via the GPT pipeline with a mock fallback.

The shipping ``/suggest/lyrics`` endpoint and the refine path's
``lyric_generator`` op both converge on
``pipeline.route_request("lyric", ...)``. This module is the shared engine
wrapper: it normalises the mode, calls the pipeline, coerces the parsed
payload into :class:`LyricSuggestion` records, and falls back to the
deterministic mock builder when the GPT path is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from backend.logging_config import get_logger
from backend.mock_pipeline import build_lyric_suggestions as _mock_build_lyric_suggestions
from ml.gpt.pipeline import GPTPipeline
from ml.gpt.prompts.lyrics import VALID_LYRIC_MODES, normalize_lyric_mode
from shared.schemas import LyricSuggestion, SessionState

LOGGER = get_logger("backend.lyric_responder")

LyricSource = Literal["gpt", "fallback"]

_DEFAULT_GPT_MODE = "Poetic"


@dataclass(frozen=True)
class LyricSuggestionsResult:
    """Lyric suggestions plus the metadata needed to audit them."""

    suggestions: list[LyricSuggestion]
    source: LyricSource
    mode: str
    model_version: str | None
    cache_hit: bool
    latency_ms: float | None
    error: str | None


def generate_lyric_suggestions(
    state: SessionState,
    *,
    mode: str,
    num_suggestions: int,
    pipeline: GPTPipeline | None = None,
    extra_instruction: str | None = None,
) -> LyricSuggestionsResult:
    """Return GPT-generated lyric suggestions, falling back to the mock on failure."""
    requested = max(1, int(num_suggestions))
    normalized_mode = _normalize_mode_or_default(mode)

    if pipeline is None:
        return _fallback(state, requested_mode=mode, num_suggestions=requested, reason="no_gpt_pipeline")

    try:
        result = pipeline.route_request(
            "lyric",
            state,
            extra_instruction or "",
            mode=normalized_mode,
            num_suggestions=requested,
        )
    except Exception as exc:
        LOGGER.info(
            "lyric_suggest_gpt_failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
            mode=normalized_mode,
            num_suggestions=requested,
        )
        return _fallback(
            state,
            requested_mode=mode,
            num_suggestions=requested,
            reason=f"transport:{exc.__class__.__name__}",
        )

    suggestions = coerce_lyric_suggestions(result.parsed, requested, fallback_mode=normalized_mode)
    if not suggestions:
        LOGGER.info(
            "lyric_suggest_empty_payload",
            payload_type=type(result.parsed).__name__,
            mode=normalized_mode,
        )
        return _fallback(
            state,
            requested_mode=mode,
            num_suggestions=requested,
            reason="empty_payload",
        )

    LOGGER.info(
        "lyric_suggest_gpt_succeeded",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        suggestion_count=len(suggestions),
    )
    return LyricSuggestionsResult(
        suggestions=suggestions,
        source="gpt",
        mode=normalized_mode,
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        error=None,
    )


def coerce_lyric_suggestions(
    parsed: Any,
    requested: int,
    *,
    fallback_mode: str = _DEFAULT_GPT_MODE,
) -> list[LyricSuggestion]:
    """Convert the GPT lyric payload into ``LyricSuggestion`` records.

    The lyric prompt returns ``{"mode", "lyrics": [...], "syllable_counts": [...]}``.
    Cached results round-trip through JSON, so ``parsed`` may also arrive as a
    JSON string. Returns ``[]`` on unparseable input so callers can fall back.
    """
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            text = parsed.strip()
            if not text:
                return []
            return [LyricSuggestion(text=text, mode=fallback_mode, syllable_count=0)]
    if not isinstance(parsed, dict):
        return []

    mode = str(parsed.get("mode") or fallback_mode)
    lines = parsed.get("lyrics") or []
    counts = parsed.get("syllable_counts") or []
    if not isinstance(lines, list):
        return []

    suggestions: list[LyricSuggestion] = []
    for index, line in enumerate(lines[:requested]):
        text = str(line).strip()
        if not text:
            continue
        syllable_count = 0
        if index < len(counts):
            try:
                syllable_count = max(0, int(counts[index]))
            except (TypeError, ValueError):
                syllable_count = 0
        suggestions.append(LyricSuggestion(text=text, mode=mode, syllable_count=syllable_count))
    return suggestions


def _normalize_mode_or_default(mode: str) -> str:
    try:
        return normalize_lyric_mode(mode)
    except ValueError:
        LOGGER.info(
            "lyric_suggest_unknown_mode",
            requested_mode=mode,
            using_default=_DEFAULT_GPT_MODE,
            valid_modes=list(VALID_LYRIC_MODES),
        )
        return _DEFAULT_GPT_MODE


def _fallback(
    state: SessionState,
    *,
    requested_mode: str,
    num_suggestions: int,
    reason: str,
) -> LyricSuggestionsResult:
    mock_suggestions = _mock_build_lyric_suggestions(state, requested_mode, num_suggestions)
    return LyricSuggestionsResult(
        suggestions=mock_suggestions,
        source="fallback",
        mode=requested_mode,
        model_version=None,
        cache_hit=False,
        latency_ms=None,
        error=reason,
    )
