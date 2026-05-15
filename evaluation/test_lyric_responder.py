"""Tests for backend.lyric_responder: GPT lyric route with mock fallback."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from backend.lyric_responder import (
    LyricSuggestionsResult,
    coerce_lyric_suggestions,
    generate_lyric_suggestions,
)
from ml.gpt.pipeline import GPTPipelineResult
from shared.schemas import LyricSuggestion, SessionState


def _session() -> SessionState:
    state = SessionState(session_id=uuid4())
    state.lyrics_text = "Hold the line"
    return state


class _FakeLyricPipeline:
    def __init__(self, *, parsed: Any = None, content: str = "{}", exc: Exception | None = None) -> None:
        self._parsed = parsed
        self._content = content
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def route_request(self, use_case: str, state: SessionState, instruction: str, **kwargs: Any) -> GPTPipelineResult:
        self.calls.append({"use_case": use_case, "instruction": instruction, **kwargs})
        if self._exc is not None:
            raise self._exc
        return GPTPipelineResult(
            use_case=use_case,
            content=self._content,
            parsed=self._parsed,
            cache_hit=False,
            provider="yandex",
            model_version="test-lyric-model",
            latency_ms=22.0,
            prompt_token_count=80,
            completion_token_count=24,
            eval_mode=False,
        )


def _gpt_payload(lines: list[str], counts: list[int] | None = None, mode: str = "Poetic") -> dict[str, Any]:
    return {
        "mode": mode,
        "lyrics": lines,
        "syllable_counts": counts if counts is not None else [len(line.split()) for line in lines],
        "rationale": "fixture",
    }


def test_gpt_lyric_payload_returns_suggestions_with_metadata() -> None:
    parsed = _gpt_payload(["Under the silver tide", "We hold the morning close"], [6, 7])
    pipeline = _FakeLyricPipeline(parsed=parsed, content=json.dumps(parsed))

    result = generate_lyric_suggestions(_session(), mode="Poetic", num_suggestions=2, pipeline=pipeline)

    assert isinstance(result, LyricSuggestionsResult)
    assert result.source == "gpt"
    assert result.mode == "Poetic"
    assert result.model_version == "test-lyric-model"
    assert result.latency_ms == 22.0
    assert len(result.suggestions) == 2
    assert result.suggestions[0].text == "Under the silver tide"
    assert result.suggestions[0].syllable_count == 6
    assert pipeline.calls[0]["use_case"] == "lyric"
    assert pipeline.calls[0]["mode"] == "Poetic"
    assert pipeline.calls[0]["num_suggestions"] == 2


def test_unknown_mode_falls_back_to_poetic_default_for_gpt() -> None:
    parsed = _gpt_payload(["Line one"], [2])
    pipeline = _FakeLyricPipeline(parsed=parsed, content=json.dumps(parsed))

    result = generate_lyric_suggestions(_session(), mode="continue", num_suggestions=1, pipeline=pipeline)

    assert result.source == "gpt"
    assert pipeline.calls[0]["mode"] == "Poetic"
    assert result.mode == "Poetic"


def test_case_insensitive_canonical_mode_is_normalized() -> None:
    parsed = _gpt_payload(["Catchy line one"], [3])
    pipeline = _FakeLyricPipeline(parsed=parsed, content=json.dumps(parsed))

    result = generate_lyric_suggestions(_session(), mode="catchy", num_suggestions=1, pipeline=pipeline)

    assert pipeline.calls[0]["mode"] == "Catchy"
    assert result.mode == "Catchy"


def test_empty_gpt_payload_falls_back_to_mock() -> None:
    pipeline = _FakeLyricPipeline(parsed={"mode": "Poetic", "lyrics": []}, content="{}")

    result = generate_lyric_suggestions(_session(), mode="Poetic", num_suggestions=3, pipeline=pipeline)

    assert result.source == "fallback"
    assert result.error == "empty_payload"
    assert len(result.suggestions) == 3  # mock always produces 3 templates


def test_transport_error_falls_back_to_mock() -> None:
    pipeline = _FakeLyricPipeline(exc=RuntimeError("connection refused"))

    result = generate_lyric_suggestions(_session(), mode="Poetic", num_suggestions=2, pipeline=pipeline)

    assert result.source == "fallback"
    assert result.error == "transport:RuntimeError"
    assert len(result.suggestions) == 2


def test_no_pipeline_uses_mock_immediately() -> None:
    result = generate_lyric_suggestions(_session(), mode="continue", num_suggestions=3, pipeline=None)

    assert result.source == "fallback"
    assert result.error == "no_gpt_pipeline"
    assert len(result.suggestions) == 3
    # Mock preserves the requested mode label rather than canonicalising it.
    assert result.mode == "continue"


def test_string_payload_is_parsed_as_json() -> None:
    payload = json.dumps(_gpt_payload(["One line"], [2]))
    pipeline = _FakeLyricPipeline(parsed=payload, content=payload)

    result = generate_lyric_suggestions(_session(), mode="Poetic", num_suggestions=1, pipeline=pipeline)

    assert result.source == "gpt"
    assert result.suggestions[0].text == "One line"


def test_coerce_lyric_suggestions_caps_at_requested_count() -> None:
    parsed = _gpt_payload(["a", "b", "c", "d"], [1, 1, 1, 1])
    suggestions = coerce_lyric_suggestions(parsed, requested=2)
    assert [s.text for s in suggestions] == ["a", "b"]


def test_coerce_lyric_suggestions_skips_empty_lines() -> None:
    parsed = {"mode": "Poetic", "lyrics": ["", "  ", "real line"], "syllable_counts": [0, 0, 2]}
    suggestions = coerce_lyric_suggestions(parsed, requested=3)
    assert [s.text for s in suggestions] == ["real line"]


def test_coerce_lyric_suggestions_handles_bad_syllable_count_types() -> None:
    parsed = {"mode": "Poetic", "lyrics": ["a"], "syllable_counts": ["not-a-number"]}
    suggestions = coerce_lyric_suggestions(parsed, requested=1)
    assert suggestions[0].syllable_count == 0


def test_coerce_lyric_suggestions_returns_empty_for_non_dict() -> None:
    assert coerce_lyric_suggestions(42, requested=1) == []
    assert coerce_lyric_suggestions(None, requested=1) == []


def test_coerce_lyric_suggestions_falls_back_to_text_for_unparseable_string() -> None:
    suggestions = coerce_lyric_suggestions("plain prose, no JSON", requested=1)
    assert len(suggestions) == 1
    assert suggestions[0].text == "plain prose, no JSON"
    assert suggestions[0].mode == "Poetic"
