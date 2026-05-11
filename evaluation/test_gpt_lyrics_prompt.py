"""Tests for GPT lyric prompt templates."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from ml.gpt.context_builder import build_session_context
from ml.gpt.prompts.lyrics import VALID_LYRIC_MODES, build_lyric_prompt, normalize_lyric_mode
from shared.schemas import (
    ChordAnnotation,
    ChordProgression,
    ChordSymbolDerivation,
    EmotionVector,
    MelodyProfile,
    SessionState,
)


def _session_state() -> SessionState:
    return SessionState(
        session_id=uuid4(),
        melody_profile=MelodyProfile(
            interval_histogram=[0.2, 0.3, 0.5],
            rhythmic_density=1.25,
            pitch_range=(57, 69),
            contour="rising",
        ),
        chord_progressions=[
            ChordProgression(
                chords=["C", "Am", "F", "G"],
                score=0.9,
                harmonic_function="tonic lift",
                explanation="Clear pop cadence.",
                chord_annotations=[
                    _annotation(position=0, symbol="C", roman_numeral="I"),
                    _annotation(position=1, symbol="Am", roman_numeral="vi"),
                    _annotation(position=2, symbol="F", roman_numeral="IV"),
                    _annotation(position=3, symbol="G", roman_numeral="V"),
                ],
            )
        ],
        lyrics_text="Hold the line\nUnder city lights",
        emotion_vector=EmotionVector(valence=0.5, arousal=0.7),
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


def _user_payload(mode: str = "Poetic") -> dict:
    context = build_session_context("lyric", _session_state(), syllable_targets_per_line=[4, 5])
    prompt = build_lyric_prompt(context, mode=mode, num_suggestions=2, extra_instruction="Keep it nocturnal.")
    return json.loads(prompt.messages[1]["content"])


def test_lyric_prompt_supports_all_three_modes() -> None:
    context = build_session_context("lyric", _session_state(), syllable_targets_per_line=[4, 5])

    for mode in VALID_LYRIC_MODES:
        prompt = build_lyric_prompt(context, mode=mode)
        payload = json.loads(prompt.messages[1]["content"])
        assert prompt.mode == mode
        assert payload["mode"] == mode
        assert mode in payload["mode_guidance"] or payload["mode_guidance"]


def test_lyric_prompt_includes_named_context_fields_and_syllable_constraints() -> None:
    payload = _user_payload()

    assert payload["required_context_fields"] == [
        "melody_profile",
        "chord_progressions",
        "syllable_targets_per_line",
        "previous_lyrics",
        "emotion_vector",
    ]
    assert payload["melody_profile"]["contour"] == "rising"
    assert payload["syllable_targets_per_line"] == [4, 5]
    assert payload["previous_lyrics"] == "Hold the line\nUnder city lights"
    assert payload["emotion_vector"] == {"valence": 0.5, "arousal": 0.7}
    assert payload["extra_instruction"] == "Keep it nocturnal."


def test_lyric_prompt_includes_chord_progressions_with_roman_numerals() -> None:
    payload = _user_payload("Catchy")

    progression = payload["chord_progressions"][0]
    assert progression["chords"] == ["C", "Am", "F", "G"]
    assert progression["roman_numerals"] == ["I", "vi", "IV", "V"]
    assert progression["chord_annotations"][1]["roman_numeral"] == "vi"


def test_lyric_prompt_system_message_requires_json_and_syllable_targets() -> None:
    context = build_session_context("lyric", _session_state(), syllable_targets_per_line=[4, 5])
    prompt = build_lyric_prompt(context, mode="Simpler")

    system_message = prompt.messages[0]["content"]
    assert "Return only valid JSON" in system_message
    assert "syllable target" in system_message


def test_lyric_prompt_requires_lyric_context() -> None:
    context = build_session_context("explain", _session_state(), last_user_question="Why?")

    with pytest.raises(ValueError, match="lyric SessionContext"):
        build_lyric_prompt(context, mode="Poetic")


def test_normalize_lyric_mode_accepts_common_casing_and_rejects_unknown() -> None:
    assert normalize_lyric_mode("poetic") == "Poetic"
    assert normalize_lyric_mode(" CATCHY ") == "Catchy"

    with pytest.raises(ValueError, match="Simpler, Poetic, Catchy"):
        normalize_lyric_mode("ambient")
