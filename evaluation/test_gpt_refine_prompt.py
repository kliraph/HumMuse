"""Tests for GPT refinement parsing prompts."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from ml.gpt.context_builder import SessionContextBuilder
from ml.gpt.prompts.refine import FEW_SHOT_EXAMPLES, build_refinement_prompt
from shared.schemas import (
    ChatMessage,
    ChordAnnotation,
    ChordProgression,
    ChordSymbolDerivation,
    EmotionVector,
    MelodyProfile,
    RefinementOp,
    RefinementPlan,
    SessionState,
)


def _session_state() -> SessionState:
    previous_plan = RefinementPlan(
        operations=[
            RefinementOp(
                target="harmonizer",
                params={"emotion_valence": -0.4},
                rationale="Make the verse sadder.",
            )
        ],
        interpretation="Made the verse sadder.",
    )
    return SessionState(
        session_id=uuid4(),
        melody_profile=MelodyProfile(
            interval_histogram=[0.1, 0.2, 0.7],
            rhythmic_density=1.4,
            pitch_range=(60, 72),
            contour="arch",
        ),
        chord_progressions=[
            ChordProgression(
                chords=["C", "Am", "F", "G"],
                score=0.89,
                harmonic_function="tonic to dominant",
                explanation="Stable cadence.",
                chord_annotations=[
                    _annotation(position=0, symbol="C", roman_numeral="I"),
                    _annotation(position=1, symbol="Am", roman_numeral="vi"),
                ],
            )
        ],
        lyrics_text="Hold the line\nUnder city lights",
        emotion_vector=EmotionVector(valence=-0.2, arousal=0.5),
        chat_history=[
            ChatMessage(role="user", content="Make the verse sadder.", timestamp="t1"),
            ChatMessage(role="assistant", content="I lowered valence and preferred minor harmony.", timestamp="t2"),
        ],
        user_params={
            "last_refinement": "Make the verse sadder.",
            "last_refinement_target": "chords",
            "last_refinement_plan": previous_plan.model_dump(),
            "last_refinement_interpretation": previous_plan.interpretation,
        },
    )


def _annotation(*, position: int, symbol: str, roman_numeral: str) -> ChordAnnotation:
    root = symbol[0]
    derivation = ChordSymbolDerivation(
        symbol=symbol,
        root=root,
        root_pc=0,
        quality="major",
        bass=root,
        bass_pc=0,
        pitch_classes=[0, 4, 7],
        inversion=0,
        roman_numeral=roman_numeral,
        harmonic_function="tonic",
        confidence=0.9,
    )
    return ChordAnnotation(
        position=position,
        symbol=symbol,
        root=root,
        quality="major",
        bass=root,
        roman_numeral=roman_numeral,
        function_label="tonic",
        alignment_percentage=0.82,
        template_phrase="stable harmonic color",
        derivation=derivation,
    )


def _prompt_payload() -> dict:
    context = SessionContextBuilder(recent_chat_messages=2).refine_context(_session_state())
    prompt = build_refinement_prompt(
        context,
        instruction="Make it less sad, then write 2 lyric options.",
        target_hint="suggestions",
    )
    return json.loads(prompt.messages[-1]["content"])


def test_refinement_few_shots_have_required_count_and_special_cases() -> None:
    assert 8 <= len(FEW_SHOT_EXAMPLES) <= 10

    multi_step = [example for example in FEW_SHOT_EXAMPLES if len(example["output"]["operations"]) > 1]
    multi_turn = [example for example in FEW_SHOT_EXAMPLES if example["chat_history"]]
    assert len(multi_step) >= 2
    assert len(multi_turn) >= 2
    assert any("грустнее" in example["instruction"] for example in FEW_SHOT_EXAMPLES)


def test_refinement_prompt_includes_chat_history_and_last_plan_results() -> None:
    payload = _prompt_payload()
    context = payload["context"]

    assert context["chat_history"] == [
        {"role": "user", "content": "Make the verse sadder.", "timestamp": "t1"},
        {
            "role": "assistant",
            "content": "I lowered valence and preferred minor harmony.",
            "timestamp": "t2",
        },
    ]
    assert context["last_refinement_plan_results"]["plan"]["operations"][0]["target"] == "harmonizer"
    assert payload["instruction"] == "Make it less sad, then write 2 lyric options."
    assert payload["target_hint"] == "suggestions"


def test_refinement_prompt_schema_requires_ordered_operations() -> None:
    context = SessionContextBuilder().refine_context(_session_state())
    prompt = build_refinement_prompt(context, instruction="make chorus sadder and shorter")

    schema = prompt.response_schema
    assert schema["required"] == ["operations", "interpretation"]
    op_schema = schema["properties"]["operations"]["items"]
    assert op_schema["required"] == ["target", "params", "rationale"]
    assert op_schema["additionalProperties"] is False
    assert op_schema["properties"]["target"]["enum"] == [
        "harmonizer",
        "melody_generator",
        "lyric_generator",
        "session",
    ]
    assert prompt.response_format["type"] == "json_schema"
    assert prompt.response_format["json_schema"]["strict"] is True
    assert prompt.response_format["json_schema"]["schema"] == prompt.response_schema


def test_refinement_prompt_messages_include_examples_and_final_payload() -> None:
    context = SessionContextBuilder().refine_context(_session_state())
    prompt = build_refinement_prompt(context, instruction="make it less sad")

    assert prompt.messages[0]["role"] == "system"
    assert "chat_history" in prompt.messages[0]["content"]
    assert len(prompt.messages) == 1 + (2 * len(FEW_SHOT_EXAMPLES)) + 1
    assert json.loads(prompt.messages[-1]["content"])["context"]["chord_progressions"][0]["roman_numerals"] == [
        "I",
        "vi",
    ]
    assert "Russian" in prompt.messages[0]["content"]


def test_refinement_prompt_validates_context_and_instruction() -> None:
    state = _session_state()
    lyric_context = SessionContextBuilder().lyric_context(state)

    with pytest.raises(ValueError, match="refine SessionContext"):
        build_refinement_prompt(lyric_context, instruction="make it brighter")

    with pytest.raises(ValueError, match="instruction"):
        build_refinement_prompt(SessionContextBuilder().refine_context(state), instruction=" ")
