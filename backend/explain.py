"""Tier 1 :class:`ExplanationReport` aggregator.

Walks the live ``SessionState`` and collects real per-module traces — DQN
chord annotations, native distributions, reward attribution, continuation
constraint logs, melody confidence, emotion modulation rationale, and the
latest refinement plan — into a single structured report. The report is
what the GPT explain dialog grounds its replies on (see
``ml/gpt/context_builder.py:explain_context`` and
``ml/gpt/tier1_formatter.py``).

Compared with the deprecated ``backend.mock_pipeline.build_explanation_report``
this aggregator preserves the *real* shapes the tier1 formatter knows how to
flatten — progression-level entries with nested ``annotations`` /
``native_distributions`` / ``chord_explanations`` — while keeping each
distribution dict compact (only the keys the formatter actually reads).
"""

from __future__ import annotations

from typing import Any

from ml.harmony.emotion_modulation import emotion_bias_rationale
from shared.schemas import (
    ChordAnnotation,
    ChordDistribution,
    ChordExplanation,
    ChordProgression,
    ExplanationReport,
    SessionState,
)


def build_explanation_report(
    state: SessionState,
    *,
    source_action: str,
    summary: str,
    use_case: str | None = None,
) -> ExplanationReport:
    """Assemble the Tier 1 :class:`ExplanationReport` for the given session."""
    return ExplanationReport(
        source_action=source_action,
        summary=summary,
        melody_confidence=_melody_confidence(state),
        chord_theory=_chord_theory(state),
        constraint_logs=_constraint_logs(state),
        emotion_mapping=_emotion_mapping(state, use_case=use_case),
        cache_status=_cache_status(state, use_case=use_case),
    )


# --- melody -----------------------------------------------------------------


def _melody_confidence(state: SessionState) -> list[dict[str, Any]]:
    return [
        {
            "pitch": note.pitch,
            "onset": note.onset,
            "duration": note.duration,
            "velocity": note.velocity,
            "confidence": note.confidence,
        }
        for note in state.melody_notes
    ]


# --- chord theory -----------------------------------------------------------


def _chord_theory(state: SessionState) -> list[dict[str, Any]]:
    return [_progression_to_chord_theory(progression) for progression in state.chord_progressions]


def _progression_to_chord_theory(progression: ChordProgression) -> dict[str, Any]:
    return {
        "chords": list(progression.chords),
        "score": progression.score,
        "model_confidence": progression.model_confidence,
        "mood_alignment": progression.mood_alignment,
        "harmonic_function": progression.harmonic_function,
        "explanation": progression.explanation,
        "annotations": [_trim_annotation(annotation) for annotation in progression.chord_annotations],
        "native_distributions": [_trim_distribution(distribution) for distribution in progression.native_distributions],
        "chord_explanations": [_trim_chord_explanation(explanation) for explanation in progression.chord_explanations],
        "native_distribution_count": len(progression.native_distributions),
    }


def _trim_annotation(annotation: ChordAnnotation) -> dict[str, Any]:
    derivation = annotation.derivation
    return {
        "position": annotation.position,
        "symbol": annotation.symbol,
        "root": annotation.root,
        "quality": annotation.quality,
        "bass": annotation.bass,
        "roman_numeral": annotation.roman_numeral,
        "function_label": annotation.function_label,
        "alignment_percentage": annotation.alignment_percentage,
        "q_margin": annotation.q_margin,
        "value_score": annotation.value_score,
        "template_phrase": annotation.template_phrase,
        "inversion": derivation.inversion,
        "pitch_classes": list(derivation.pitch_classes),
        "derivation": {
            "symbol": derivation.symbol,
            "root": derivation.root,
            "quality": derivation.quality,
            "candidate_symbols": list(derivation.candidate_symbols),
            "failure_modes": list(derivation.failure_modes),
            "confidence": derivation.confidence,
            "harmonic_function": derivation.harmonic_function,
        },
    }


def _trim_distribution(distribution: ChordDistribution) -> dict[str, Any]:
    """Keep only the keys ``tier1_formatter`` actually reads.

    Native distributions carry a lot of noise (dueling values, advantages,
    rest probabilities, noise sample counts) that would bloat the persisted
    session state. The formatter only consumes policy membership/labels,
    octave/inversion choices, reward attribution, and the emotion bias
    summary, so trim aggressively here.
    """
    policy = distribution.policy
    reward = distribution.reward_attribution
    emotion = distribution.emotion_bias
    selected = distribution.selected
    return {
        "position": distribution.position,
        "policy": {
            "pitch_class_membership": list(policy.pitch_class_membership),
            "pitch_class_labels": list(policy.pitch_class_labels),
            "octave": dict(policy.octave),
            "inversion": dict(policy.inversion),
        },
        "reward_attribution": (
            {
                "harmony_rule": reward.harmony_rule,
                "progression_penalty": reward.progression_penalty,
                "chord_tone_inclusion": reward.chord_tone_inclusion,
                "mutual_info": reward.mutual_info,
                "key_fit_proxy": reward.key_fit_proxy,
            }
            if reward is not None
            else None
        ),
        "emotion_bias": (
            {
                "applied": emotion.applied,
                "rationale": emotion.rationale,
                "q_delta_per_pc": list(emotion.q_delta_per_pc),
            }
            if emotion is not None
            else None
        ),
        "selected": {
            "is_rest": selected.is_rest,
            "octave": selected.octave,
            "inversion": selected.inversion,
            "pcs": list(selected.pcs),
            "pc_names": list(selected.pc_names),
        },
    }


def _trim_chord_explanation(explanation: ChordExplanation) -> dict[str, Any]:
    return {
        "position": explanation.position,
        "chord_symbol": explanation.chord_symbol,
        "q_chosen": explanation.q_chosen,
        "q_runner_up": explanation.q_runner_up,
        "runner_up_symbol": explanation.runner_up_symbol,
        "margin": explanation.margin,
        "value_score": explanation.value_score,
        "advantage_score": explanation.advantage_score,
        "prev_chord_link": explanation.prev_chord_link,
        "reward_components": dict(explanation.reward_components),
        "emotion_bias_summary": explanation.emotion_bias_summary,
        "prose": explanation.prose,
    }


# --- continuation constraints -----------------------------------------------


def _constraint_logs(state: SessionState) -> list[dict[str, Any]]:
    user_params = state.user_params or {}
    trace = list(user_params.get("continuation_constraint_trace") or [])
    rejection_metadata = list(user_params.get("continuation_rejection_metadata") or [])

    logs: list[dict[str, Any]] = list(trace)
    for entry in rejection_metadata:
        logs.append({**entry, "kind": "rejection_metadata"})

    if not logs:
        # Pre-continuation sessions still get *something* useful: a synthetic
        # acceptance row per current suggestion so the explain prompt isn't
        # empty when the user asks about a suggestion that's on screen.
        for index, suggestion in enumerate(state.melody_suggestions):
            logs.append(
                {
                    "candidate": index + 1,
                    "status": "accepted",
                    "reason": suggestion.explanation,
                    "score": suggestion.coherence_score,
                    "engine": suggestion.engine,
                }
            )

    summary = {
        "candidate_count": user_params.get("continuation_candidate_count"),
        "survivor_count": user_params.get("continuation_survivor_count"),
        "engines": user_params.get("continuation_engines"),
    }
    if any(value is not None for value in summary.values()):
        logs = [{"kind": "summary", **summary}, *logs]
    return logs


# --- emotion mapping --------------------------------------------------------


def _emotion_mapping(state: SessionState, *, use_case: str | None) -> dict[str, Any]:
    user_params = state.user_params or {}
    mapping: dict[str, Any] = {}
    if state.emotion_vector is not None:
        valence = float(state.emotion_vector.valence)
        arousal = float(state.emotion_vector.arousal)
        mapping["valence"] = valence
        mapping["arousal"] = arousal
        mapping["rationale"] = emotion_bias_rationale(
            valence,
            arousal,
            key=state.detected_key or "C major",
        )
    if last_plan := user_params.get("last_refinement_plan"):
        mapping["last_refinement"] = {
            "instruction": user_params.get("last_refinement"),
            "target": user_params.get("last_refinement_target"),
            "interpretation": user_params.get("last_refinement_interpretation"),
            "source": user_params.get("last_refinement_source"),
            "operations": list(last_plan.get("operations", [])),
            "execution": user_params.get("last_refinement_execution"),
        }
    if mood := user_params.get("mood"):
        mapping["session_mood"] = mood
    if genre := user_params.get("genre"):
        mapping["session_genre"] = genre
    if use_case is not None:
        mapping["explain_use_case"] = use_case
    return mapping


# --- cache status -----------------------------------------------------------


def _cache_status(state: SessionState, *, use_case: str | None) -> dict[str, Any]:
    user_params = state.user_params or {}
    status: dict[str, Any] = {}
    if use_case is not None:
        status["use_case"] = use_case
    for key in (
        "last_refinement_cache_hit",
        "last_chat_cache_hit",
        "last_lyric_cache_hit",
        "last_refinement_model_version",
        "last_chat_model_version",
        "last_lyric_model_version",
    ):
        if key in user_params:
            status[key] = user_params[key]
    return status
