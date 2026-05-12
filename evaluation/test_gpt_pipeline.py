"""Tests for unified GPT pipeline routing, caching, eval mode, and logging."""

from __future__ import annotations

import json
from uuid import uuid4

from ml.gpt.client import LLMConfig, LLMResponse
from ml.gpt.pipeline import GPTPipeline, GPTPipelineConfig
from shared.schemas import ChatMessage, ExplanationReport, SessionState


class FakeClient:
    def __init__(self, content: str) -> None:
        self.config = LLMConfig(provider="yandex", model="test-model", api_key="key")
        self.content = content
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(
            content=self.content,
            model="test-model",
            provider="yandex",
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )


class FakeCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], str] = {}
        self.get_calls: list[tuple[str, str, str]] = []
        self.put_calls: list[tuple[str, str, str, str]] = []

    def get(self, prompt_hash: str, model_version: str, use_case: str) -> str | None:
        self.get_calls.append((prompt_hash, model_version, use_case))
        return self.values.get((prompt_hash, model_version, use_case))

    def put(self, prompt_hash: str, response_json: str, model_version: str, use_case: str) -> None:
        self.put_calls.append((prompt_hash, response_json, model_version, use_case))
        self.values[(prompt_hash, model_version, use_case)] = response_json


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.events.append((event, fields))


class FakeClock:
    def __init__(self) -> None:
        self.values = iter([1.0, 1.123, 2.0, 2.004])

    def __call__(self) -> float:
        return next(self.values)


def _session_state() -> SessionState:
    return SessionState(
        session_id=uuid4(),
        explanation_report=ExplanationReport(
            source_action="test",
            summary="Tier 1 report",
            melody_confidence=[{"pitch": 62, "confidence": 0.9}],
            constraint_logs=[],
            chord_theory=[],
            emotion_mapping={"valence": 0.1, "arousal": 0.3},
        ),
        chat_history=[ChatMessage(role="user", content="Why Dm here?")],
        lyrics_text="Hold the line",
    )


def test_pipeline_logs_per_call_metadata_and_writes_cache() -> None:
    client = FakeClient(json.dumps({"lyrics": ["Hold the line"], "syllable_counts": [3], "rationale": "Fits context."}))
    cache = FakeCache()
    logger = FakeLogger()
    pipeline = GPTPipeline(
        client,
        cache=cache,
        config=GPTPipelineConfig(eval_mode=False),
        logger=logger,
        clock=FakeClock(),
    )

    result = pipeline.route_request("lyric", _session_state(), "write one line", syllable_targets_per_line=[3])

    assert result.cache_hit is False
    assert result.prompt_token_count == 11
    assert result.completion_token_count == 7
    assert result.latency_ms == 123.0
    assert len(cache.put_calls) == 1
    assert logger.events == [
        (
            "gpt_call_completed",
            {
                "provider": "yandex",
                "model_version": "test-model",
                "use_case": "lyric",
                "cache_hit": False,
                "latency_ms": 123.0,
                "prompt_token_count": 11,
                "completion_token_count": 7,
                "eval_mode": False,
            },
        )
    ]


def test_pipeline_cache_hit_skips_client_and_logs_cache_hit() -> None:
    client = FakeClient("should not be called")
    cache = FakeCache()
    logger = FakeLogger()
    first = GPTPipeline(
        FakeClient(json.dumps({"answer": "Because Dm is ii.", "evidence": [], "limits": ""})),
        cache=cache,
        config=GPTPipelineConfig(eval_mode=False),
        logger=FakeLogger(),
        clock=FakeClock(),
    )
    first.route_request("explain", _session_state(), "Why Dm here?")

    second = GPTPipeline(
        client,
        cache=cache,
        config=GPTPipelineConfig(eval_mode=False),
        logger=logger,
        clock=FakeClock(),
    )
    result = second.route_request("explain", _session_state(), "Why Dm here?")

    assert result.cache_hit is True
    assert result.content == '{"answer": "Because Dm is ii.", "evidence": [], "limits": ""}'
    assert client.calls == []
    assert logger.events[0][1]["cache_hit"] is True


def test_eval_mode_pins_temperature_seed_and_disables_refine_cache() -> None:
    response = {
        "operations": [
            {
                "target": "harmonizer",
                "params": {"emotion_valence": 0.2},
                "rationale": "Less sad.",
            }
        ],
        "interpretation": "Make it less sad.",
    }
    client = FakeClient(json.dumps(response))
    cache = FakeCache()
    logger = FakeLogger()
    pipeline = GPTPipeline(
        client,
        cache=cache,
        config=GPTPipelineConfig(eval_mode=True, eval_seed=42),
        logger=logger,
        clock=FakeClock(),
    )

    result = pipeline.route_request("refine", _session_state(), "make it less sad")

    assert result.eval_mode is True
    assert result.parsed.operations[0].target == "harmonizer"
    assert client.calls[0]["kwargs"]["temperature"] == 0
    assert client.calls[0]["kwargs"]["seed"] == 42
    assert "response_format" in client.calls[0]["kwargs"]
    assert cache.get_calls == []
    assert cache.put_calls == []
    assert logger.events[0][1]["eval_mode"] is True
    assert logger.events[0][1]["cache_hit"] is False


def test_non_eval_refine_can_use_cache() -> None:
    response = {
        "operations": [
            {
                "target": "harmonizer",
                "params": {"emotion_valence": 0.2},
                "rationale": "Less sad.",
            }
        ],
        "interpretation": "Make it less sad.",
    }
    client = FakeClient(json.dumps(response))
    cache = FakeCache()
    pipeline = GPTPipeline(
        client,
        cache=cache,
        config=GPTPipelineConfig(eval_mode=False),
        logger=None,
        clock=FakeClock(),
    )

    pipeline.route_request("refine", _session_state(), "make it less sad")

    assert len(cache.get_calls) == 1
    assert len(cache.put_calls) == 1
