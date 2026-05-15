"""Tests for backend.chat_responder: GPT explain route with mock fallback."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from backend.chat_responder import ChatReply, build_chat_reply
from ml.gpt.pipeline import GPTPipelineResult
from shared.schemas import (
    ChordProgression,
    EmotionVector,
    ExplanationReport,
    NoteEvent,
    SessionState,
)


def _session_with_report() -> SessionState:
    state = SessionState(session_id=uuid4())
    state.melody_notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
    ]
    state.detected_key = "C major"
    state.detected_tempo = 120.0
    state.emotion_vector = EmotionVector(valence=0.2, arousal=0.5)
    state.chord_progressions = [ChordProgression(chords=["C", "G", "Am", "F"], score=0.9, explanation="bright")]
    state.explanation_report = ExplanationReport(
        source_action="chords_from_lyrics",
        summary="DQN harmonizer produced bright progression.",
    )
    return state


class _FakeExplainPipeline:
    def __init__(self, *, parsed: Any = None, content: str = "{}", exc: Exception | None = None) -> None:
        self._parsed = parsed
        self._content = content
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def route_request(self, use_case: str, state: SessionState, message: str, **kwargs: Any) -> GPTPipelineResult:
        self.calls.append({"use_case": use_case, "message": message, **kwargs})
        if self._exc is not None:
            raise self._exc
        return GPTPipelineResult(
            use_case=use_case,
            content=self._content,
            parsed=self._parsed,
            cache_hit=False,
            provider="yandex",
            model_version="test-model",
            latency_ms=18.5,
            prompt_token_count=120,
            completion_token_count=40,
            eval_mode=False,
        )


def test_gpt_answer_with_limits_is_returned_verbatim() -> None:
    parsed = {
        "answer": "The progression starts on tonic and resolves cleanly to subdominant.",
        "limits": "Voice-leading specifics rely on the rendered MIDI.",
    }
    pipeline = _FakeExplainPipeline(parsed=parsed, content=json.dumps(parsed))

    reply = build_chat_reply(_session_with_report(), "Why these chords?", pipeline=pipeline)

    assert isinstance(reply, ChatReply)
    assert reply.source == "gpt"
    assert reply.message.role == "assistant"
    assert "tonic" in reply.message.content
    assert reply.limits == "Voice-leading specifics rely on the rendered MIDI."
    assert reply.model_version == "test-model"
    assert reply.latency_ms == 18.5
    assert reply.error is None
    assert pipeline.calls[0]["use_case"] == "explain"
    assert pipeline.calls[0]["message"] == "Why these chords?"


def test_gpt_answer_without_limits_returns_none_for_limits() -> None:
    parsed = {"answer": "The melody outlines the I chord."}
    pipeline = _FakeExplainPipeline(parsed=parsed, content=json.dumps(parsed))

    reply = build_chat_reply(_session_with_report(), "Explain the melody.", pipeline=pipeline)

    assert reply.source == "gpt"
    assert reply.message.content == "The melody outlines the I chord."
    assert reply.limits is None


def test_gpt_string_payload_is_parsed_as_json() -> None:
    payload = '{"answer": "C major sits firmly as the tonal home.", "limits": null}'
    pipeline = _FakeExplainPipeline(parsed=payload, content=payload)

    reply = build_chat_reply(_session_with_report(), "What key are we in?", pipeline=pipeline)

    assert reply.source == "gpt"
    assert "tonal home" in reply.message.content


def test_gpt_text_envelope_fallback_path() -> None:
    """When the pipeline returns ``{'text': '...'}`` (no JSON object), still use it."""
    parsed = {"text": "Plain prose answer about the song."}
    pipeline = _FakeExplainPipeline(parsed=parsed, content=json.dumps(parsed))

    reply = build_chat_reply(_session_with_report(), "Tell me about the song.", pipeline=pipeline)

    assert reply.source == "gpt"
    assert reply.message.content == "Plain prose answer about the song."


def test_empty_answer_falls_back_to_mock() -> None:
    parsed = {"answer": ""}
    pipeline = _FakeExplainPipeline(parsed=parsed, content="")

    reply = build_chat_reply(_session_with_report(), "What?", pipeline=pipeline)

    assert reply.source == "fallback"
    assert reply.error == "empty_answer"
    assert reply.message.role == "assistant"
    assert reply.message.content  # mock always returns something


def test_transport_error_falls_back_to_mock() -> None:
    pipeline = _FakeExplainPipeline(exc=RuntimeError("connection refused"))

    reply = build_chat_reply(_session_with_report(), "Why this melody?", pipeline=pipeline)

    assert reply.source == "fallback"
    assert reply.error == "transport:RuntimeError"
    assert reply.message.role == "assistant"
    assert reply.message.content


def test_no_pipeline_uses_mock() -> None:
    reply = build_chat_reply(_session_with_report(), "Anything?", pipeline=None)

    assert reply.source == "fallback"
    assert reply.error == "no_gpt_pipeline"
    assert reply.message.role == "assistant"
    assert "explanation" in reply.message.content.lower() or "session" in reply.message.content.lower()


def test_unparseable_string_payload_is_returned_as_prose() -> None:
    """If the model returns prose (not JSON), the responder still surfaces it."""
    pipeline = _FakeExplainPipeline(parsed="Some plain prose without JSON structure.", content="Some plain prose without JSON structure.")

    reply = build_chat_reply(_session_with_report(), "Hi", pipeline=pipeline)

    assert reply.source == "gpt"
    assert "plain prose" in reply.message.content
