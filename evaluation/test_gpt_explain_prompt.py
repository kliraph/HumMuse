"""Tests for GPT explanation prompt templates."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from ml.gpt.context_builder import build_session_context
from ml.gpt.prompts.explain import build_explanation_prompt
from shared.schemas import ChatMessage, ExplanationReport, SessionState


def _session_state() -> SessionState:
    return SessionState(
        session_id=uuid4(),
        explanation_report=ExplanationReport(
            source_action="chords_from_lyrics",
            summary="DQN chose Dm because melody and harmony evidence aligned.",
            melody_confidence=[
                {"pitch": 62, "onset": 1.0, "confidence": 0.94},
                {"pitch": 65, "onset": 2.0, "confidence": 0.91},
            ],
            constraint_logs=[
                {"candidate": 1, "status": "accepted", "reason": "fits key and density", "score": 0.87}
            ],
            chord_theory=[
                {
                    "progression": ["C", "Dm", "G"],
                    "annotations": [
                        {
                            "position": 1,
                            "symbol": "Dm",
                            "roman_numeral": "ii",
                            "function_label": "predominant",
                            "alignment_percentage": 0.82,
                        }
                    ],
                    "native_distributions": [
                        {
                            "position": 1,
                            "policy": {
                                "pitch_class_membership": [0.02, 0.01, 0.28, 0.01, 0.02, 0.24, 0.01, 0.2, 0.01, 0.02, 0.01, 0.02, 0.15],
                                "pitch_class_labels": [
                                    "C",
                                    "C#",
                                    "D",
                                    "D#",
                                    "E",
                                    "F",
                                    "F#",
                                    "G",
                                    "G#",
                                    "A",
                                    "A#",
                                    "B",
                                    "triad_sentinel",
                                ],
                            },
                            "reward_attribution": {
                                "harmony_rule": 12.0,
                                "progression_penalty": 0.0,
                                "chord_tone_inclusion": 1.0,
                                "mutual_info": 0.2,
                                "key_fit_proxy": 1.0,
                            },
                        }
                    ],
                }
            ],
            emotion_mapping={"valence": -0.2, "arousal": 0.4},
            cache_status={"use_case": "explain", "cache_hit": False},
        ),
        chat_history=[
            ChatMessage(role="user", content="Can you explain the harmony?", timestamp="t1"),
            ChatMessage(role="assistant", content="Of course — happy to walk through it.", timestamp="t2"),
            ChatMessage(role="user", content="Why Dm here?", timestamp="t3"),
        ],
    )


def _payload(question: str | None = None) -> dict:
    context = build_session_context("explain", _session_state())
    prompt = build_explanation_prompt(context, question=question)
    return json.loads(prompt.messages[1]["content"])


def test_explanation_prompt_includes_session_context_question_and_history() -> None:
    payload = _payload()

    assert payload["context_provided"] == [
        "session_abc",
        "tier1_natural_language",
        "explanation_report_tier_1",
        "chat_history",
        "last_user_question",
    ]
    assert payload["question"] == "Why Dm here?"
    assert payload["explanation_report_tier_1"]["summary"].startswith("DQN chose Dm")
    assert len(payload["chat_history"]) == 3
    assert payload["last_user_question"] == "Why Dm here?"


def test_explanation_prompt_surfaces_session_abc_and_tier1_prose() -> None:
    payload = _payload("Why Dm here?")

    assert isinstance(payload["session_abc"], str)
    assert payload["session_abc"].startswith("X:1\n")
    prose = payload["tier1_natural_language"]
    assert "SUMMARY: DQN chose Dm" in prose
    assert "CHORD DECISIONS (CLSTM four-head distributions + DQN reward attributions):" in prose
    assert "Position 1: Dm" in prose
    assert "(ii)" in prose


def test_explanation_prompt_omits_citation_enforcement_fields() -> None:
    payload = _payload("Why Dm here?")

    assert "evidence_to_prefer" not in payload
    assert "required_grounding_fields" not in payload
    assert "required_output" not in payload


def test_explanation_system_prompt_uses_teacher_voice_and_avoids_citation_pressure() -> None:
    context = build_session_context("explain", _session_state())
    prompt = build_explanation_prompt(context)
    system = prompt.messages[0]["content"]

    assert "music-teacher" in system or "music teacher" in system
    assert "suitability" in system or "feels right" in system or "why the choice" in system
    assert "Plain language first" in system
    assert "BACHI" in system  # still explicitly excluded
    assert "Q-values" in system  # explicitly discouraged from numeric citations
    assert "ground every claim" not in system
    assert "cite" not in system.lower()


def test_explanation_prompt_provides_minimal_response_format() -> None:
    context = build_session_context("explain", _session_state())
    prompt = build_explanation_prompt(context)

    assert prompt.response_schema["required"] == ["answer"]
    assert "evidence" not in prompt.response_schema["properties"]
    assert "limits" in prompt.response_schema["properties"]
    assert prompt.response_format["type"] == "json_schema"
    assert prompt.response_format["json_schema"]["strict"] is True
    assert prompt.response_format["json_schema"]["schema"] == prompt.response_schema


def test_explanation_prompt_validates_context_and_question() -> None:
    state = _session_state()
    lyric_context = build_session_context("lyric", state)

    with pytest.raises(ValueError, match="explain SessionContext"):
        build_explanation_prompt(lyric_context, question="Why?")

    empty_question_context = build_session_context("explain", SessionState(session_id=uuid4()))
    with pytest.raises(ValueError, match="question"):
        build_explanation_prompt(empty_question_context)
