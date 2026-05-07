"""Deterministic explanation builders for DQN chord decisions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from ml.harmony.chord_symbols import PC_NAMES, QUALITY_TEMPLATES
from shared.schemas import ChordDistribution, ChordExplanation, ChordSymbolDerivation


def build_explanation(distribution: ChordDistribution, derivation: ChordSymbolDerivation) -> ChordExplanation:
    """Build the UI-facing explanation object for one selected chord."""

    q_chosen = _chosen_q(distribution)
    q_runner_up, runner_up_symbol = _runner_up(distribution)
    margin = _selected_margin(distribution)
    value_score = distribution.dueling.value_pc
    advantage_score = _chosen_advantage(distribution)
    reward_components = _reward_components(distribution)
    emotion_bias_summary = _emotion_bias_summary(distribution)
    prev_chord_link = distribution.context.prev_chord_symbol
    explanation = ChordExplanation(
        position=distribution.position,
        chord_symbol=derivation.symbol,
        q_chosen=q_chosen,
        q_runner_up=q_runner_up,
        runner_up_symbol=runner_up_symbol,
        margin=margin,
        value_score=value_score,
        advantage_score=advantage_score,
        prev_chord_link=prev_chord_link,
        reward_components=reward_components,
        emotion_bias_summary=emotion_bias_summary,
        prose="pending",
    )
    return explanation.model_copy(update={"prose": prose_template(explanation)})


def prose_template(explanation: ChordExplanation) -> str:
    """Render a deterministic, compact explanation sentence."""

    prev = f" after {explanation.prev_chord_link}" if explanation.prev_chord_link else ""
    emotion = f" Emotion: {explanation.emotion_bias_summary}" if explanation.emotion_bias_summary else ""
    rewards = ", ".join(
        f"{name}={value:.2f}"
        for name, value in sorted(explanation.reward_components.items())
    )
    reward_text = f" Rewards: {rewards}." if rewards else ""
    return (
        f"{explanation.chord_symbol}{prev} was selected with Q={explanation.q_chosen:.2f} "
        f"over {explanation.runner_up_symbol} at Q={explanation.q_runner_up:.2f} "
        f"(margin {explanation.margin:.2f}); value={explanation.value_score:.2f}, "
        f"advantage={explanation.advantage_score:.2f}.{reward_text}{emotion}"
    )


def _chosen_q(distribution: ChordDistribution) -> float:
    selected = distribution.selected
    if selected.is_rest:
        return float(distribution.q_values.rest)
    values = [
        _at(distribution.q_values.octave, (selected.octave or 2) - 2),
        _at(distribution.q_values.inversion, selected.inversion or 0),
    ]
    values.extend(_at(distribution.q_values.pitch_class, pc) for pc in selected.pcs)
    return sum(values) / len(values) if values else 0.0


def _runner_up(distribution: ChordDistribution) -> tuple[float, str]:
    if distribution.selected.is_rest:
        return _best_pitch_set_alternative(distribution, chord_size=3, excluded=None)

    chosen = tuple(sorted({int(pc) % 12 for pc in distribution.selected.pcs}))
    if not chosen:
        return _best_pitch_set_alternative(distribution, chord_size=3, excluded=None)

    return _best_pitch_set_alternative(distribution, chord_size=len(chosen), excluded=chosen)


def _best_pitch_set_alternative(
    distribution: ChordDistribution,
    *,
    chord_size: int,
    excluded: tuple[int, ...] | None,
) -> tuple[float, str]:
    candidates = []
    for pcs in combinations(range(12), max(1, min(4, chord_size))):
        if excluded is not None and pcs == excluded:
            continue
        candidates.append((_selection_q(distribution, pcs), _pitch_set_symbol(pcs)))
    if not candidates:
        return 0.0, "none"
    value, symbol = max(candidates, key=lambda item: (item[0], item[1]))
    return float(value), symbol


def _selection_q(distribution: ChordDistribution, pcs: tuple[int, ...]) -> float:
    selected = distribution.selected
    octave_index = _best_index(distribution.q_values.octave) if selected.is_rest else (selected.octave or 2) - 2
    inversion_index = _best_index(distribution.q_values.inversion) if selected.is_rest else (selected.inversion or 0)
    values = [
        _at(distribution.q_values.octave, octave_index),
        _at(distribution.q_values.inversion, inversion_index),
    ]
    values.extend(_at(distribution.q_values.pitch_class, pc) for pc in pcs)
    return sum(values) / len(values) if values else 0.0


def _pitch_set_symbol(pcs: tuple[int, ...]) -> str:
    pc_set = set(pcs)
    for root_pc in range(12):
        for template in QUALITY_TEMPLATES:
            template_pcs = {(root_pc + interval) % 12 for interval in template.intervals}
            if template_pcs == pc_set:
                return f"{PC_NAMES[root_pc]}{template.suffix}"
    return "-".join(PC_NAMES[pc] for pc in pcs)


def _selected_margin(distribution: ChordDistribution) -> float:
    if distribution.selected.is_rest:
        return float(distribution.q_margin.rest)
    return min(
        float(distribution.q_margin.octave),
        float(distribution.q_margin.inversion),
        float(distribution.q_margin.pitch_class),
    )


def _chosen_advantage(distribution: ChordDistribution) -> float:
    selected = distribution.selected
    if selected.is_rest:
        return _at(distribution.dueling.advantage_rest, 0)
    values = [
        _at(distribution.dueling.advantage_octave, (selected.octave or 2) - 2),
        _at(distribution.dueling.advantage_inversion, selected.inversion or 0),
    ]
    values.extend(_at(distribution.dueling.advantage_pc, pc) for pc in selected.pcs)
    return sum(values) / len(values) if values else 0.0


def _reward_components(distribution: ChordDistribution) -> dict[str, float]:
    if distribution.reward_attribution is None:
        return {}
    return {
        "harmony_rule": distribution.reward_attribution.harmony_rule,
        "progression_penalty": distribution.reward_attribution.progression_penalty,
        "chord_tone_inclusion": distribution.reward_attribution.chord_tone_inclusion,
        "mutual_info": distribution.reward_attribution.mutual_info,
        "key_fit_proxy": distribution.reward_attribution.key_fit_proxy,
    }


def _emotion_bias_summary(distribution: ChordDistribution) -> str | None:
    if distribution.emotion_bias is None or not distribution.emotion_bias.applied:
        return None
    nonzero = [
        value
        for value in distribution.emotion_bias.q_delta_per_pc
        if abs(value) > 1e-12
    ]
    total = sum(nonzero)
    return f"{distribution.emotion_bias.rationale} Net pitch-class Q delta={total:.2f}."


def _best_index(values: list[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda index: values[index])


def _at(values: list[float], index: int) -> float:
    if not values:
        return 0.0
    return float(values[max(0, min(len(values) - 1, index))])
