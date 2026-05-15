"""Build assistant chat replies grounded in Tier 1 explanation data.

Wraps :class:`ml.gpt.pipeline.GPTPipeline` for the ``explain`` use case. The
prompt itself is already wired to consume ``session_abc`` + Tier 1 prose +
the raw ``ExplanationReport`` via
:func:`ml.gpt.context_builder.build_session_context`, so this module only
deals with: pipeline plumbing, response coercion, and falling back to the
deterministic mock reply when the GPT path is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from backend.logging_config import get_logger
from backend.mock_pipeline import build_chat_reply as _mock_build_chat_reply
from ml.gpt.pipeline import GPTPipeline
from shared.schemas import ChatMessage, SessionState

LOGGER = get_logger("backend.chat_responder")

ReplySource = Literal["gpt", "fallback"]


@dataclass(frozen=True)
class ChatReply:
    """Assistant chat reply plus the metadata needed to audit it."""

    message: ChatMessage
    source: ReplySource
    model_version: str | None
    cache_hit: bool
    latency_ms: float | None
    limits: str | None
    error: str | None


def build_chat_reply(
    state: SessionState,
    message: str,
    *,
    pipeline: GPTPipeline | None = None,
) -> ChatReply:
    """Return a grounded assistant reply, falling back to the mock on any failure."""
    if pipeline is None:
        return _fallback(state, message, reason="no_gpt_pipeline")

    try:
        result = pipeline.route_request("explain", state, message)
    except Exception as exc:
        LOGGER.info(
            "chat_explain_gpt_failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
            question=message,
        )
        return _fallback(state, message, reason=f"transport:{exc.__class__.__name__}")

    answer, limits = _extract_answer_and_limits(result.parsed, fallback_content=result.content)
    if not answer:
        LOGGER.info(
            "chat_explain_empty_answer",
            payload_type=type(result.parsed).__name__,
            question=message,
        )
        return _fallback(state, message, reason="empty_answer")

    LOGGER.info(
        "chat_explain_gpt_succeeded",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        has_limits=bool(limits),
    )
    return ChatReply(
        message=ChatMessage(role="assistant", content=answer, timestamp=_now_iso()),
        source="gpt",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        limits=limits,
        error=None,
    )


def _extract_answer_and_limits(parsed: Any, *, fallback_content: str) -> tuple[str, str | None]:
    """Pull ``answer`` / ``limits`` out of the GPT payload regardless of shape.

    The explain prompt asks for a strict ``{"answer", "limits"}`` JSON object,
    but cached results round-trip through JSON so ``parsed`` may already be a
    dict, a JSON string, or — if the model returned plain prose — a
    ``{"text": "..."}`` envelope from the pipeline's text fallback.
    """
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return parsed.strip(), None
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        limits = parsed.get("limits")
        if isinstance(answer, str) and answer.strip():
            return answer.strip(), (limits.strip() if isinstance(limits, str) and limits.strip() else None)
        text = parsed.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip(), None
    if fallback_content and fallback_content.strip():
        return fallback_content.strip(), None
    return "", None


def _fallback(state: SessionState, message: str, *, reason: str) -> ChatReply:
    mock_message = _mock_build_chat_reply(state, message)
    return ChatReply(
        message=mock_message,
        source="fallback",
        model_version=None,
        cache_hit=False,
        latency_ms=None,
        limits=None,
        error=reason,
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
