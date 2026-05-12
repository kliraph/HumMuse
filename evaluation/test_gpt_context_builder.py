"""Tests for GPT session context assembly."""

from __future__ import annotations

from uuid import uuid4

from ml.gpt.context_builder import SessionContextBuilder, build_session_context
from shared.schemas import (
    ChatMessage,
    ChordAnnotation,
    ChordProgression,
    ChordSymbolDerivation,
    EmotionVector,
    ExplanationReport,
    MelodyProfile,
    RefinementOp,
    RefinementPlan,
    SessionState,
)


def _session_state() -> SessionState:
    refinement_plan = RefinementPlan(
        operations=[
            RefinementOp(
                target="harmonizer",
                params={"chord_complexity": "high"},
                rationale="Make the harmony richer.",
            )
        ],
        interpretation="Make the harmony richer.",
    )
    return SessionState(
        session_id=uuid4(),
        melody_profile=MelodyProfile(
            interval_histogram=[0.1, 0.2, 0.7],
            rhythmic_density=1.5,
            pitch_range=(60, 72),
            contour="arch",
        ),
        chord_progressions=[
            ChordProgression(
                chords=["C", "Am", "F", "G"],
                score=0.88,
                model_confidence=0.91,
                mood_alignment=0.72,
                harmonic_function="tonic to dominant",
                explanation="Stable pop progression.",
                chord_annotations=[
                    _annotation(position=0, symbol="C", roman_numeral="I"),
                    _annotation(position=1, symbol="Am", roman_numeral="vi"),
                ],
            )
        ],
        lyrics_text="Hold the line\nUnder city lights",
        emotion_vector=EmotionVector(valence=0.4, arousal=0.6),
        explanation_report=ExplanationReport(
            source_action="chords_from_lyrics",
            summary="Tier 1 explanation summary.",
            melody_confidence=[{"pitch": 60, "confidence": 0.95}],
            constraint_logs=[{"candidate": 1, "status": "accepted"}],
            chord_theory=[{"progression": ["C", "Am"], "roman": ["I", "vi"]}],
            emotion_mapping={"valence": 0.4, "arousal": 0.6},
            cache_status={"use_case": "explain", "cache_hit": False},
        ),
        chat_history=[
            ChatMessage(role="user", content="Can it be brighter?", timestamp="t1"),
            ChatMessage(role="assistant", content="Yes, I can lift the harmony.", timestamp="t2"),
            ChatMessage(role="user", content="Why C here?", timestamp="t3"),
        ],
        user_params={
            "last_refinement": "make it jazzier",
            "last_refinement_target": "chords",
            "last_refinement_plan": refinement_plan.model_dump(),
            "last_refinement_interpretation": refinement_plan.interpretation,
        },
    )


def _annotation(*, position: int, symbol: str, roman_numeral: str) -> ChordAnnotation:
    derivation = ChordSymbolDerivation(
        symbol=symbol,
        root=symbol[0],
        root_pc=0,
        quality="major",
        bass=symbol[0],
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
        root=symbol[0],
        quality="major",
        bass=symbol[0],
        roman_numeral=roman_numeral,
        function_label="tonic",
        alignment_percentage=0.8,
        q_margin=0.3,
        value_score=1.2,
        template_phrase="stable tonic color",
        derivation=derivation,
    )


def test_lyric_context_contains_explicit_named_fields() -> None:
    context = build_session_context("lyric", _session_state(), syllable_targets_per_line=[4, 5])

    assert context.use_case == "lyric"
    assert set(context.fields) == {
        "melody_profile",
        "chord_progressions",
        "syllable_targets_per_line",
        "previous_lyrics",
        "emotion_vector",
    }
    assert context["melody_profile"]["contour"] == "arch"
    assert context["chord_progressions"][0]["roman_numerals"] == ["I", "vi"]
    assert context["syllable_targets_per_line"] == [4, 5]
    assert context["previous_lyrics"] == "Hold the line\nUnder city lights"
    assert context["emotion_vector"] == {"valence": 0.4, "arousal": 0.6}
    assert '"roman_numerals"' in context.to_prompt_text()


def test_refine_context_contains_lyric_fields_recent_chat_and_refinement_results() -> None:
    context = SessionContextBuilder(recent_chat_messages=2).refine_context(_session_state())

    for name in (
        "melody_profile",
        "chord_progressions",
        "syllable_targets_per_line",
        "previous_lyrics",
        "emotion_vector",
        "recent_chat_history",
        "last_refinement_plan_results",
    ):
        assert name in context
    assert [message["content"] for message in context["recent_chat_history"]] == [
        "Yes, I can lift the harmony.",
        "Why C here?",
    ]
    assert context["last_refinement_plan_results"]["instruction"] == "make it jazzier"
    assert context["last_refinement_plan_results"]["plan"]["operations"][0]["target"] == "harmonizer"


def test_explain_context_contains_tier_1_report_chat_history_and_last_question() -> None:
    context = build_session_context("explain", _session_state())

    assert set(context.fields) == {
        "session_abc",
        "tier1_natural_language",
        "explanation_report_tier_1",
        "chat_history",
        "last_user_question",
    }
    assert context["explanation_report_tier_1"]["summary"] == "Tier 1 explanation summary."
    assert len(context["chat_history"]) == 3
    assert context["last_user_question"] == "Why C here?"


def test_explain_context_accepts_explicit_last_user_question() -> None:
    context = build_session_context("explain", _session_state(), last_user_question="Why the vi chord?")

    assert context["last_user_question"] == "Why the vi chord?"


def test_explain_context_includes_session_abc_and_tier1_natural_language() -> None:
    context = build_session_context("explain", _session_state())

    abc = context["session_abc"]
    assert isinstance(abc, str)
    assert abc.startswith("X:1\n")
    assert "K:Cmaj" in abc or "K:" in abc

    summary = context["tier1_natural_language"]
    assert isinstance(summary, str)
    assert "SOURCE: chords_from_lyrics" in summary
    assert "Tier 1 explanation summary." in summary
