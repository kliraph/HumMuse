"""Format Tier-1 XAI data into ChatMusician-style analytic prose.

The SFT YandexGPT Lite (fine-tuned on ChatMusician's SFT corpus) expects
score-grounded analysis in natural-language form. This module converts the
structured `ExplanationReport` into prose the model can cite from.

Chord decisions are framed as "CLSTM four-head distributions + DQN reward
attributions" — the policy network is `CLSTM_Chord_MH`, trained via DQN.

Expected keys per chord-decision entry in `ExplanationReport.chord_theory`
(populate via `format_chord_theory_from_progression` when assembling the
report so the formatter has consistent keys to read):

  position, symbol (or chord_symbol), roman_numeral, function_label,
  pitch_class_membership, pitch_class_labels, q_chosen, q_runner_up,
  runner_up_symbol, margin (or q_margin), reward_components,
  alignment_percentage, inversion, octave_choice, template_phrase, prose,
  prev_chord_link, pitch_classes.

Progression-level entries (with nested `annotations` / `native_distributions`)
are also accepted and flattened transparently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from shared.schemas import ChordProgression, ExplanationReport

_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def format_tier1_summary(report: ExplanationReport | None) -> str:
    """Render an `ExplanationReport` as analytic prose for the explanation prompt."""
    if report is None:
        return "No Tier-1 explanation data available for this session yet."

    sections: list[str] = [
        f"SOURCE: {report.source_action}",
        f"SUMMARY: {report.summary}",
    ]
    if report.melody_confidence:
        sections.append(_format_melody_confidence(report.melody_confidence))
    chord_section = _format_chord_decisions(report.chord_theory)
    if chord_section:
        sections.append(chord_section)
    if report.constraint_logs:
        sections.append(_format_constraint_logs(report.constraint_logs))
    emotion_section = _format_emotion_mapping(report.emotion_mapping)
    if emotion_section:
        sections.append(emotion_section)
    return "\n\n".join(sections)


def format_chord_theory_from_progression(progression: ChordProgression) -> list[dict[str, Any]]:
    """Produce well-shaped per-position chord-decision dicts from a `ChordProgression`.

    Use this when assembling `ExplanationReport.chord_theory` (Phase 7.4) so the
    natural-language formatter reads consistent keys.
    """
    annotations_by_pos = {a.position: a for a in progression.chord_annotations}
    distributions_by_pos = {d.position: d for d in progression.native_distributions}
    explanations_by_pos = {e.position: e for e in progression.chord_explanations}
    positions = sorted(
        set(annotations_by_pos) | set(distributions_by_pos) | set(explanations_by_pos)
    )
    entries: list[dict[str, Any]] = []
    for position in positions:
        entry: dict[str, Any] = {"position": position}
        annotation = annotations_by_pos.get(position)
        distribution = distributions_by_pos.get(position)
        explanation = explanations_by_pos.get(position)
        if annotation is not None:
            entry.update(
                symbol=annotation.symbol,
                roman_numeral=annotation.roman_numeral,
                function_label=annotation.function_label,
                alignment_percentage=annotation.alignment_percentage,
                q_margin=annotation.q_margin,
                value_score=annotation.value_score,
                template_phrase=annotation.template_phrase,
                inversion=annotation.derivation.inversion,
                pitch_classes=list(annotation.derivation.pitch_classes),
            )
        if distribution is not None:
            entry.update(
                pitch_class_membership=list(distribution.policy.pitch_class_membership),
                pitch_class_labels=list(distribution.policy.pitch_class_labels),
                octave_choice=_argmax_label(distribution.policy.octave),
                inversion_distribution=dict(distribution.policy.inversion),
            )
            if distribution.reward_attribution is not None:
                entry["reward_components"] = {
                    "harmony_rule": distribution.reward_attribution.harmony_rule,
                    "progression_penalty": distribution.reward_attribution.progression_penalty,
                    "chord_tone_inclusion": distribution.reward_attribution.chord_tone_inclusion,
                    "mutual_info": distribution.reward_attribution.mutual_info,
                    "key_fit_proxy": distribution.reward_attribution.key_fit_proxy,
                }
        if explanation is not None:
            entry.update(
                q_chosen=explanation.q_chosen,
                q_runner_up=explanation.q_runner_up,
                runner_up_symbol=explanation.runner_up_symbol,
                margin=explanation.margin,
                prev_chord_link=explanation.prev_chord_link,
                prose=explanation.prose,
            )
        entries.append(entry)
    return entries


def _format_melody_confidence(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["MELODY (Basic Pitch per-note confidence):"]
    for row in rows:
        pitch = row.get("pitch")
        confidence = row.get("confidence")
        onset = row.get("onset")
        pitch_str = _midi_to_pitch_name(pitch)
        confidence_str = _format_probability(confidence)
        onset_str = f" at beat {float(onset):.2f}" if isinstance(onset, (int, float)) else ""
        lines.append(f"  - {pitch_str}{onset_str}: confidence {confidence_str}")
    return "\n".join(lines)


def _format_chord_decisions(rows: Sequence[Mapping[str, Any]]) -> str:
    flat = list(_flatten_chord_theory(rows))
    if not flat:
        return ""
    lines = ["CHORD DECISIONS (CLSTM four-head distributions + DQN reward attributions):"]
    for entry in flat:
        lines.extend(_format_one_chord_decision(entry))
    return "\n".join(lines)


def _flatten_chord_theory(rows: Sequence[Mapping[str, Any]]):
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        nested_annotations = row.get("annotations")
        nested_distributions = row.get("native_distributions")
        nested_explanations = row.get("chord_explanations")
        if any(
            isinstance(group, Sequence)
            for group in (nested_annotations, nested_distributions, nested_explanations)
        ):
            yield from _merge_progression_level_entry(
                annotations=nested_annotations or [],
                distributions=nested_distributions or [],
                explanations=nested_explanations or [],
                progression_chords=row.get("progression") or [],
            )
        else:
            yield row


def _merge_progression_level_entry(
    *,
    annotations: Sequence[Mapping[str, Any]],
    distributions: Sequence[Mapping[str, Any]],
    explanations: Sequence[Mapping[str, Any]],
    progression_chords: Sequence[str],
):
    by_position: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        position = int(annotation.get("position", 0))
        entry = by_position.setdefault(position, {"position": position})
        for key in (
            "symbol", "roman_numeral", "function_label",
            "alignment_percentage", "q_margin", "value_score",
            "template_phrase", "inversion", "pitch_classes",
        ):
            if key in annotation:
                entry[key] = annotation[key]
    for distribution in distributions:
        position = int(distribution.get("position", 0))
        entry = by_position.setdefault(position, {"position": position})
        policy = distribution.get("policy") or {}
        if isinstance(policy, Mapping):
            if "pitch_class_membership" in policy:
                entry["pitch_class_membership"] = list(policy["pitch_class_membership"])
            if "pitch_class_labels" in policy:
                entry["pitch_class_labels"] = list(policy["pitch_class_labels"])
            if isinstance(policy.get("octave"), Mapping):
                entry["octave_choice"] = _argmax_label(policy["octave"])
            if isinstance(policy.get("inversion"), Mapping):
                entry["inversion_distribution"] = dict(policy["inversion"])
        reward = distribution.get("reward_attribution")
        if isinstance(reward, Mapping):
            entry["reward_components"] = dict(reward)
    for explanation in explanations:
        position = int(explanation.get("position", 0))
        entry = by_position.setdefault(position, {"position": position})
        for key in (
            "q_chosen", "q_runner_up", "runner_up_symbol",
            "margin", "prev_chord_link", "prose",
        ):
            if key in explanation:
                entry[key] = explanation[key]
    for position in sorted(by_position):
        entry = by_position[position]
        if "symbol" not in entry and 0 <= position < len(progression_chords):
            entry["symbol"] = progression_chords[position]
        yield entry


def _format_one_chord_decision(row: Mapping[str, Any]) -> list[str]:
    position = row.get("position", "?")
    symbol = row.get("symbol") or row.get("chord_symbol") or "?"
    roman = row.get("roman_numeral") or row.get("roman")
    function = row.get("function_label")
    header_parts = [f"Position {position}: {symbol}"]
    if roman:
        header_parts.append(f"({roman})")
    if function:
        header_parts.append(f"— {function}")
    lines = [f"  - {' '.join(header_parts)}"]

    membership_str = _top_pitch_classes(
        row.get("pitch_class_membership"),
        row.get("pitch_class_labels"),
    )
    if membership_str:
        lines.append(f"    pitch-class membership: {membership_str}")

    voicing_bits: list[str] = []
    inversion = row.get("inversion")
    if inversion is not None:
        voicing_bits.append(f"inversion {inversion}")
    octave_choice = row.get("octave_choice")
    if octave_choice is not None:
        voicing_bits.append(f"octave choice {octave_choice}")
    pitch_classes = row.get("pitch_classes")
    if isinstance(pitch_classes, Sequence) and pitch_classes:
        named = ", ".join(_PITCH_CLASS_NAMES[int(pc) % 12] for pc in pitch_classes)
        voicing_bits.append(f"chord tones [{named}]")
    if voicing_bits:
        lines.append(f"    voicing: {', '.join(voicing_bits)}")

    q_line = _format_q_value_line(row)
    if q_line:
        lines.append(q_line)

    rewards = row.get("reward_components")
    if isinstance(rewards, Mapping) and rewards:
        reward_str = ", ".join(f"{name} {float(value):+.2f}" for name, value in rewards.items())
        lines.append(f"    DQN reward attribution: {reward_str}")

    alignment = row.get("alignment_percentage")
    if isinstance(alignment, (int, float)):
        lines.append(f"    strong-beat chord-tone alignment: {_format_probability(alignment)}")

    template = row.get("template_phrase")
    if template:
        lines.append(f"    theory note: {template}")

    prose = row.get("prose")
    if prose:
        lines.append(f"    decision narrative: {prose}")

    prev_link = row.get("prev_chord_link")
    if prev_link:
        lines.append(f"    voice leading from previous chord: {prev_link}")

    return lines


def _format_q_value_line(row: Mapping[str, Any]) -> str | None:
    q_chosen = row.get("q_chosen")
    if not isinstance(q_chosen, (int, float)):
        return None
    q_runner_up = row.get("q_runner_up")
    runner_up_symbol = row.get("runner_up_symbol")
    margin = row.get("margin") or row.get("q_margin")
    parts = [f"DQN Q-value chosen: {float(q_chosen):.2f}"]
    if isinstance(q_runner_up, (int, float)) and runner_up_symbol:
        parts.append(f"runner-up {runner_up_symbol} (Q={float(q_runner_up):.2f})")
    if isinstance(margin, (int, float)):
        parts.append(f"margin {float(margin):.2f}")
    return f"    {', '.join(parts)}"


def _format_constraint_logs(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["CONTINUATION CONSTRAINTS:"]
    for row in rows:
        candidate = row.get("candidate")
        status = row.get("status") or "n/a"
        reason = row.get("reason") or row.get("rule")
        score = row.get("score") if isinstance(row.get("score"), (int, float)) else row.get("value")
        prefix = f"candidate {candidate}: {status}" if candidate is not None else str(status)
        bits = [prefix]
        if reason:
            bits.append(f"rule={reason}")
        if isinstance(score, (int, float)):
            bits.append(f"score={float(score):.2f}")
        lines.append(f"  - {' '.join(bits)}")
    return "\n".join(lines)


def _format_emotion_mapping(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return ""
    lines = ["EMOTION MAPPING:"]
    valence = mapping.get("valence")
    arousal = mapping.get("arousal")
    if isinstance(valence, (int, float)) and isinstance(arousal, (int, float)):
        lines.append(f"  - valence={float(valence):+.2f}, arousal={float(arousal):+.2f}")
    applied_temperature = mapping.get("applied_temperature")
    if isinstance(applied_temperature, (int, float)):
        lines.append(f"  - sampling temperature applied: {float(applied_temperature):.2f}")
    applied_bias = mapping.get("applied_bias") or mapping.get("vocabulary_bias")
    if isinstance(applied_bias, (int, float)):
        lines.append(f"  - vocabulary bias applied: {float(applied_bias):+.2f}")
    rationale = mapping.get("rationale")
    if rationale:
        lines.append(f"  - rationale: {rationale}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _format_probability(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "n/a"


def _argmax_label(distribution: Mapping[str, Any]) -> str | None:
    if not distribution:
        return None
    candidates = [(k, float(v)) for k, v in distribution.items() if isinstance(v, (int, float))]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _top_pitch_classes(membership: Any, labels: Any, top_k: int = 4) -> str:
    if not isinstance(membership, Sequence) or not isinstance(labels, Sequence):
        return ""
    pairs: list[tuple[str, float]] = []
    for index, value in enumerate(membership):
        if not isinstance(value, (int, float)):
            continue
        if index >= len(labels):
            continue
        label = labels[index]
        if not isinstance(label, str) or label == "triad_sentinel":
            continue
        pairs.append((label, float(value)))
    if not pairs:
        return ""
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    top = pairs[:top_k]
    return "{" + ", ".join(f"{label}:{value:.2f}" for label, value in top) + "}"


def _midi_to_pitch_name(midi_value: Any) -> str:
    if not isinstance(midi_value, (int, float)) or isinstance(midi_value, bool):
        return str(midi_value)
    midi_int = int(midi_value)
    if not 0 <= midi_int <= 127:
        return str(midi_value)
    pc = midi_int % 12
    octave = midi_int // 12 - 1
    return f"{_PITCH_CLASS_NAMES[pc]}{octave}"
