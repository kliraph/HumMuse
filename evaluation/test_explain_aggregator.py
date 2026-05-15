"""Tests for backend.explain.build_explanation_report (Tier 1 aggregator)."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from backend.explain import build_explanation_report
from ml.gpt.tier1_formatter import format_tier1_summary
from shared.schemas import (
    ChordAnnotation,
    ChordContext,
    ChordContextNote,
    ChordDistribution,
    ChordDueling,
    ChordEmotionBias,
    ChordExplanation,
    ChordPolicy,
    ChordProgression,
    ChordQMargin,
    ChordQValues,
    ChordRewardAttribution,
    ChordSymbolDerivation,
    EmotionVector,
    ExplanationReport,
    MelodySuggestion,
    NoteEvent,
    SelectedChord,
    SessionState,
)

_PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel"]


def _derivation(symbol: str = "C", root: str = "C") -> ChordSymbolDerivation:
    return ChordSymbolDerivation(
        symbol=symbol,
        root=root,
        root_pc=0,
        quality="major",
        bass=root,
        bass_pc=0,
        pitch_classes=[0, 4, 7],
        inversion=0,
        roman_numeral="I",
        harmonic_function="tonic",
        confidence=0.95,
        candidate_symbols=[symbol, "C/E"],
        failure_modes=[],
    )


def _annotation(position: int = 0, symbol: str = "C") -> ChordAnnotation:
    return ChordAnnotation(
        position=position,
        symbol=symbol,
        root=symbol,
        quality="major",
        bass=symbol,
        roman_numeral="I",
        function_label="tonic",
        alignment_percentage=0.8,
        q_margin=0.42,
        value_score=0.31,
        template_phrase="tonic stability",
        derivation=_derivation(symbol=symbol, root=symbol),
    )


def _distribution(position: int = 0) -> ChordDistribution:
    membership = [0.0] * 13
    membership[0] = 0.9
    membership[4] = 0.6
    membership[7] = 0.5
    return ChordDistribution(
        position=position,
        policy=ChordPolicy(
            rest={"rest": 0.05},
            octave={"2": 0.3, "3": 0.7},
            inversion={str(idx): 0.25 for idx in range(4)},
            pitch_class={label: value for label, value in zip(_PITCH_CLASS_LABELS, membership)},
            pitch_class_membership=membership,
            pitch_class_labels=_PITCH_CLASS_LABELS,
        ),
        q_values=ChordQValues(
            rest=0.05,
            octave=[0.3, 0.7],
            inversion=[0.25, 0.20, 0.15, 0.10],
            pitch_class=membership,
        ),
        q_margin=ChordQMargin(rest=0.1, octave=0.4, inversion=0.05, pitch_class=0.3),
        dueling=ChordDueling(
            value_rest=0.0,
            value_octave=0.5,
            value_inversion=0.2,
            value_pc=0.6,
            advantage_rest=[0.0],
            advantage_octave=[-0.2, 0.2],
            advantage_inversion=[0.0, 0.0, 0.0, 0.0],
            advantage_pc=[0.0] * 13,
        ),
        context=ChordContext(
            prev_chord_pcs=[],
            prev_chord_symbol=None,
            current_note=ChordContextNote(pitch=60, duration=4, position=0),
        ),
        reward_attribution=ChordRewardAttribution(
            harmony_rule=0.4,
            progression_penalty=-0.05,
            chord_tone_inclusion=0.6,
            mutual_info=0.1,
            key_fit_proxy=0.5,
        ),
        emotion_bias=ChordEmotionBias(
            applied=True,
            q_delta_per_pc=[0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            rationale="Major-triad bias around tonic 0.",
        ),
        selected=SelectedChord(is_rest=False, octave=3, inversion=0, pcs=[0, 4, 7], pc_names=["C", "E", "G"]),
    )


def _chord_explanation(position: int = 0, chord_symbol: str = "C") -> ChordExplanation:
    return ChordExplanation(
        position=position,
        chord_symbol=chord_symbol,
        q_chosen=0.62,
        q_runner_up=0.45,
        runner_up_symbol="Am",
        margin=0.17,
        value_score=0.6,
        advantage_score=0.05,
        prev_chord_link=None,
        reward_components={"harmony_rule": 0.4, "chord_tone_inclusion": 0.6},
        emotion_bias_summary="Major-triad bias.",
        prose="C was chosen with Q=0.62 over Am at Q=0.45 (margin 0.17).",
    )


def _full_progression() -> ChordProgression:
    return ChordProgression(
        chords=["C", "G", "Am", "F"],
        score=0.92,
        model_confidence=0.88,
        mood_alignment=0.81,
        harmonic_function="I–V–vi–IV",
        explanation="Bright pop progression with tonic anchor.",
        chord_annotations=[_annotation(position=index, symbol=symbol) for index, symbol in enumerate(["C", "G", "Am", "F"])],
        native_distributions=[_distribution(position=index) for index in range(4)],
        chord_explanations=[_chord_explanation(position=index, chord_symbol=symbol) for index, symbol in enumerate(["C", "G", "Am", "F"])],
    )


def _session_with_real_traces() -> SessionState:
    state = SessionState(session_id=uuid4())
    state.melody_notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=98, confidence=0.91),
    ]
    state.detected_key = "C major"
    state.detected_tempo = 120.0
    state.emotion_vector = EmotionVector(valence=-0.3, arousal=0.4)
    state.chord_progressions = [_full_progression()]
    state.user_params.update(
        {
            "continuation_constraint_trace": [
                {"candidate": 1, "status": "accepted", "checks": {"chord_tone": True}, "score": 0.81},
                {"candidate": 2, "status": "rejected", "checks": {"chord_tone": False}, "rule": "chord_tone_alignment"},
            ],
            "continuation_rejection_metadata": [
                {"model_id": "seed7", "temperature": 0.9, "rejection_reason": "out_of_key", "candidate_idx": 3}
            ],
            "continuation_candidate_count": 8,
            "continuation_survivor_count": 5,
            "continuation_engines": ["music_transformer"],
            "last_refinement": "make it darker",
            "last_refinement_target": "chords",
            "last_refinement_plan": {
                "operations": [
                    {
                        "target": "harmonizer",
                        "params": {"emotion_valence": -0.4, "prefer_minor_color": True},
                        "rationale": "Darker harmony.",
                    }
                ],
                "interpretation": "Reharmonize toward darker color.",
            },
            "last_refinement_interpretation": "Reharmonize toward darker color.",
            "last_refinement_source": "gpt",
            "last_refinement_execution": {
                "results": [
                    {
                        "target": "harmonizer",
                        "status": "applied",
                        "rationale": "Darker harmony.",
                        "changes": {"progression_count": 1},
                        "error": None,
                    }
                ]
            },
            "last_refinement_cache_hit": False,
            "last_chat_cache_hit": True,
            "last_lyric_cache_hit": False,
            "mood": "melancholic",
            "genre": "indie folk",
        }
    )
    return state


# --- structural / serialization guarantees -----------------------------------


def test_report_is_pydantic_validated_and_json_serialisable() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="session_chat", summary="Tier 1 grounded reply.", use_case="explain")
    assert isinstance(report, ExplanationReport)
    json.dumps(report.model_dump())  # raises on any non-serialisable leaf


def test_source_action_and_summary_are_propagated() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="chords_from_lyrics", summary="DQN ran.", use_case=None)
    assert report.source_action == "chords_from_lyrics"
    assert report.summary == "DQN ran."


# --- melody confidence --------------------------------------------------------


def test_melody_confidence_includes_duration_and_velocity() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y")
    row = report.melody_confidence[0]
    assert row["pitch"] == 60
    assert row["duration"] == 1.0
    assert row["velocity"] == 96
    assert row["confidence"] == pytest.approx(0.95)


def test_melody_confidence_empty_when_no_notes() -> None:
    state = SessionState(session_id=uuid4())
    report = build_explanation_report(state, source_action="x", summary="y")
    assert report.melody_confidence == []


# --- chord theory -------------------------------------------------------------


def test_chord_theory_keeps_existing_progression_envelope_shape() -> None:
    """The earlier mock emitted per-progression envelopes; keep that shape so
    callers asserting on chord_theory[0]['native_distribution_count'] keep working.
    """
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y")
    entry = report.chord_theory[0]
    assert entry["chords"] == ["C", "G", "Am", "F"]
    assert entry["native_distribution_count"] == 4
    assert entry["annotations"][0]["symbol"] == "C"
    assert entry["annotations"][0]["roman_numeral"] == "I"


def test_chord_theory_distributions_are_trimmed_to_formatter_keys() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y")
    distribution = report.chord_theory[0]["native_distributions"][0]
    # Keep what tier1_formatter reads.
    assert "pitch_class_membership" in distribution["policy"]
    assert "pitch_class_labels" in distribution["policy"]
    assert distribution["reward_attribution"]["harmony_rule"] == pytest.approx(0.4)
    assert distribution["emotion_bias"]["rationale"].startswith("Major-triad")
    # Drop heavy internals so the report stays compact.
    assert "dueling" not in distribution
    assert "noise_samples" not in distribution
    assert "q_margin" not in distribution
    assert "q_values" not in distribution


def test_chord_theory_explanations_preserve_q_values_and_prose() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y")
    explanation = report.chord_theory[0]["chord_explanations"][0]
    assert explanation["q_chosen"] == pytest.approx(0.62)
    assert explanation["runner_up_symbol"] == "Am"
    assert "Q=0.62" in explanation["prose"]


def test_chord_theory_empty_when_no_progressions() -> None:
    state = SessionState(session_id=uuid4())
    report = build_explanation_report(state, source_action="x", summary="y")
    assert report.chord_theory == []


# --- constraint logs ----------------------------------------------------------


def test_constraint_logs_merges_trace_rejection_and_summary() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y")
    logs = report.constraint_logs
    summary = logs[0]
    assert summary["kind"] == "summary"
    assert summary["candidate_count"] == 8
    assert summary["survivor_count"] == 5
    # Trace entries follow, then rejection_metadata tagged with kind.
    candidates = [row.get("candidate") for row in logs if "candidate" in row]
    assert 1 in candidates
    assert 2 in candidates
    rejection = next(row for row in logs if row.get("kind") == "rejection_metadata")
    assert rejection["rejection_reason"] == "out_of_key"


def test_constraint_logs_falls_back_to_melody_suggestions_when_no_trace() -> None:
    state = SessionState(session_id=uuid4())
    state.melody_suggestions = [
        MelodySuggestion(
            midi_bytes=b"\x00",
            notes=[NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=80, confidence=0.9)],
            explanation="Fixture continuation",
            coherence_score=0.7,
            engine="music_transformer",
        )
    ]
    report = build_explanation_report(state, source_action="x", summary="y")
    assert len(report.constraint_logs) == 1
    assert report.constraint_logs[0]["status"] == "accepted"
    assert report.constraint_logs[0]["engine"] == "music_transformer"


def test_constraint_logs_empty_when_nothing_to_say() -> None:
    state = SessionState(session_id=uuid4())
    report = build_explanation_report(state, source_action="x", summary="y")
    assert report.constraint_logs == []


# --- emotion mapping ----------------------------------------------------------


def test_emotion_mapping_includes_rationale_and_last_refinement() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="session_chat", summary="y", use_case="explain")
    mapping = report.emotion_mapping
    assert mapping["valence"] == pytest.approx(-0.3)
    assert mapping["arousal"] == pytest.approx(0.4)
    assert mapping["rationale"]
    assert mapping["last_refinement"]["instruction"] == "make it darker"
    assert mapping["last_refinement"]["source"] == "gpt"
    assert mapping["last_refinement"]["operations"][0]["target"] == "harmonizer"
    assert mapping["session_mood"] == "melancholic"
    assert mapping["session_genre"] == "indie folk"
    assert mapping["explain_use_case"] == "explain"


def test_emotion_mapping_empty_when_no_emotion_or_refinement() -> None:
    state = SessionState(session_id=uuid4())
    report = build_explanation_report(state, source_action="x", summary="y")
    assert report.emotion_mapping == {}


# --- cache status -------------------------------------------------------------


def test_cache_status_surfaces_recent_use_case_flags() -> None:
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="x", summary="y", use_case="explain")
    cache = report.cache_status
    assert cache["use_case"] == "explain"
    assert cache["last_refinement_cache_hit"] is False
    assert cache["last_chat_cache_hit"] is True
    assert cache["last_lyric_cache_hit"] is False


def test_cache_status_omits_unset_keys() -> None:
    state = SessionState(session_id=uuid4())
    report = build_explanation_report(state, source_action="x", summary="y", use_case="explain")
    cache = report.cache_status
    assert cache == {"use_case": "explain"}


# --- tier1 formatter compatibility -------------------------------------------


def test_tier1_formatter_renders_aggregator_output() -> None:
    """Round-trip: the formatter must accept the aggregator's shapes and
    surface DQN Q-values, reward attribution, and emotion mapping in prose.
    """
    state = _session_with_real_traces()
    report = build_explanation_report(state, source_action="session_chat", summary="Grounded reply.", use_case="explain")
    prose = format_tier1_summary(report)
    assert "CHORD DECISIONS" in prose
    assert "C (I)" in prose
    assert "DQN Q-value chosen: 0.62" in prose
    assert "DQN reward attribution" in prose
    assert "EMOTION MAPPING" in prose
    assert "valence=-0.30" in prose
