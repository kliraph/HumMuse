"""Tests for the Tier-1 natural-language formatter used in the explanation prompt."""

from __future__ import annotations

from ml.gpt.tier1_formatter import format_chord_theory_from_progression, format_tier1_summary
from shared.schemas import (
    ChordAnnotation,
    ChordDistribution,
    ChordExplanation,
    ChordProgression,
    ChordSymbolDerivation,
    ExplanationReport,
    SelectedChord,
)


_FIXTURE_PCM = [0.02, 0.01, 0.94, 0.01, 0.02, 0.91, 0.01, 0.01, 0.01, 0.88, 0.01, 0.02, 0.15]
_FIXTURE_PCM_LABELS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel",
]


def _flat_chord_theory_report() -> ExplanationReport:
    return ExplanationReport(
        source_action="chords_from_lyrics",
        summary="DQN chose Dm because melody and harmony evidence aligned.",
        melody_confidence=[
            {"pitch": 62, "onset": 1.0, "confidence": 0.94},
            {"pitch": 65, "onset": 2.0, "confidence": 0.91},
        ],
        constraint_logs=[
            {"candidate": 1, "status": "accepted", "reason": "fits key and density", "score": 0.87},
            {"candidate": 2, "status": "rejected", "reason": "out of key"},
        ],
        chord_theory=[
            {
                "position": 1,
                "symbol": "Dm",
                "roman_numeral": "ii",
                "function_label": "predominant",
                "alignment_percentage": 0.82,
                "pitch_class_membership": _FIXTURE_PCM,
                "pitch_class_labels": _FIXTURE_PCM_LABELS,
                "octave_choice": "3",
                "inversion": 0,
                "q_chosen": 1.42,
                "q_runner_up": 1.11,
                "runner_up_symbol": "F",
                "margin": 0.31,
                "reward_components": {
                    "harmony_rule": 12.0,
                    "progression_penalty": -0.05,
                    "chord_tone_inclusion": 1.0,
                    "mutual_info": 0.2,
                    "key_fit_proxy": 1.0,
                },
                "template_phrase": "predominant ii — common pre-dominant in pop progressions",
            }
        ],
        emotion_mapping={
            "valence": -0.2,
            "arousal": 0.4,
            "applied_temperature": 0.8,
            "applied_bias": -0.3,
            "rationale": "low valence biases minor pitch-class memberships",
        },
    )


def test_format_tier1_summary_renders_all_sections() -> None:
    summary = format_tier1_summary(_flat_chord_theory_report())

    assert "SOURCE: chords_from_lyrics" in summary
    assert "SUMMARY: DQN chose Dm" in summary
    assert "MELODY (Basic Pitch per-note confidence):" in summary
    assert "D4" in summary
    assert "F4" in summary
    assert "CHORD DECISIONS (CLSTM four-head distributions + DQN reward attributions):" in summary
    assert "Position 1: Dm (ii) — predominant" in summary
    assert "pitch-class membership: {D:0.94, F:0.91, A:0.88" in summary
    assert "DQN Q-value chosen: 1.42, runner-up F (Q=1.11), margin 0.31" in summary
    assert "harmony_rule +12.00" in summary
    assert "strong-beat chord-tone alignment: 0.82" in summary
    assert "CONTINUATION CONSTRAINTS:" in summary
    assert "candidate 1: accepted" in summary
    assert "candidate 2: rejected" in summary
    assert "EMOTION MAPPING:" in summary
    assert "valence=-0.20, arousal=+0.40" in summary
    assert "sampling temperature applied: 0.80" in summary


def test_format_tier1_summary_handles_none_report() -> None:
    assert "No Tier-1 explanation data" in format_tier1_summary(None)


def test_format_tier1_summary_flattens_nested_progression_entries() -> None:
    report = ExplanationReport(
        source_action="chords_from_lyrics",
        summary="Progression-level fixture",
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
                            "pitch_class_membership": _FIXTURE_PCM,
                            "pitch_class_labels": _FIXTURE_PCM_LABELS,
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
    )

    summary = format_tier1_summary(report)

    assert "Position 1: Dm (ii) — predominant" in summary
    assert "pitch-class membership: {D:0.94, F:0.91, A:0.88" in summary
    assert "harmony_rule +12.00" in summary


def test_format_chord_theory_from_progression_emits_one_entry_per_position() -> None:
    progression = ChordProgression(
        chords=["C", "Dm"],
        score=0.9,
        explanation="fixture",
        chord_annotations=[
            ChordAnnotation(
                position=0,
                symbol="C",
                root="C",
                quality="major",
                bass="C",
                roman_numeral="I",
                function_label="tonic",
                alignment_percentage=0.9,
                q_margin=0.4,
                value_score=1.0,
                template_phrase="stable tonic color",
                derivation=ChordSymbolDerivation(
                    symbol="C",
                    root="C",
                    root_pc=0,
                    quality="major",
                    bass="C",
                    bass_pc=0,
                    pitch_classes=[0, 4, 7],
                    inversion=0,
                    roman_numeral="I",
                    harmonic_function="tonic",
                    confidence=0.95,
                ),
            )
        ],
        native_distributions=[
            ChordDistribution(
                position=0,
                policy={
                    "rest": {"rest": 0.05},
                    "octave": {"3": 0.8, "4": 0.2},
                    "inversion": {"0": 0.9, "1": 0.05, "2": 0.03, "3": 0.02},
                    "pitch_class": {"C": 0.95, "E": 0.9, "G": 0.85},
                    "pitch_class_membership": [0.95] + [0.0] * 12,
                    "pitch_class_labels": _FIXTURE_PCM_LABELS,
                },
                selected=SelectedChord(
                    is_rest=False,
                    octave=3,
                    inversion=0,
                    pcs=[0, 4, 7],
                    pc_names=["C", "E", "G"],
                ),
            )
        ],
        chord_explanations=[
            ChordExplanation(
                position=0,
                chord_symbol="C",
                q_chosen=2.10,
                q_runner_up=1.50,
                runner_up_symbol="Am",
                margin=0.60,
                value_score=1.0,
                advantage_score=0.4,
                prev_chord_link="opens the progression on tonic",
                reward_components={"harmony_rule": 8.0, "key_fit_proxy": 1.0},
                prose="Tonic anchor that establishes the key.",
            )
        ],
    )

    entries = format_chord_theory_from_progression(progression)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["position"] == 0
    assert entry["symbol"] == "C"
    assert entry["roman_numeral"] == "I"
    assert entry["function_label"] == "tonic"
    assert entry["q_chosen"] == 2.10
    assert entry["runner_up_symbol"] == "Am"
    assert entry["pitch_class_membership"][0] == 0.95
    assert entry["octave_choice"] == "3"
    assert entry["pitch_classes"] == [0, 4, 7]
