"""Pure reward-attribution helpers for DQN chord decisions.

These functions mirror the interpretable portions of RL-Chord's training
reward from ``train_utils.py`` without importing training-time torch losses or
mutual-information classifiers.
"""

from __future__ import annotations

from typing import Any

from ml.harmony.chord_symbols import QUALITY_TEMPLATES, parse_key

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
BASIC_SPACE_DISTANCE_SCALE = 12.0


def harmony_rule(chosen_chord: Any, melody_event: tuple[int, int, int] | None, prev_chord: Any, key: str | None) -> float:
    """Interval consonance between melody/chord plus consonance inside the chord."""

    chosen = _coerce_chord(chosen_chord)
    if chosen["is_rest"]:
        return 0.0
    chord_pcs = chosen["pcs"]
    if not chord_pcs:
        return 0.0

    melody_pc = _melody_pc(melody_event)
    score_between = 0.0
    if melody_pc is not None:
        score_between = sum(_harmonic_level(pc, melody_pc) for pc in chord_pcs) / len(chord_pcs)

    score_self = 0.0
    pair_count = 0
    for left_index, left_pc in enumerate(chord_pcs):
        for right_pc in chord_pcs[left_index + 1:]:
            score_self += _harmonic_level(left_pc, right_pc)
            pair_count += 1
    if pair_count:
        score_self /= pair_count
    return round(score_between + score_self, 6)


def progression_penalty(chosen_chord: Any, melody_event: tuple[int, int, int] | None, prev_chord: Any, key: str | None) -> float:
    """Repetition term plus normalized Lerdahl basic-space distance."""

    chosen = _coerce_chord(chosen_chord)
    previous = _coerce_chord(prev_chord)
    if previous is None:
        return 0.0

    position = melody_event[2] if melody_event is not None else 0
    repetition = 0.0
    if _same_chord(chosen, previous):
        repetition = -1.0 if position == 0 else 1.0

    superstrong = 0.0
    if not chosen["is_rest"] and not previous["is_rest"]:
        distance = lerdahl_basic_space_distance(set(previous["pcs"]), set(chosen["pcs"]), key)
        superstrong = -(distance / BASIC_SPACE_DISTANCE_SCALE)
    return round(repetition + superstrong, 6)


def lerdahl_basic_space_distance(prev_pcs: set[int], chord_pcs: set[int], key: str | None) -> int:
    """Lerdahl-style delta over two embedded chord basic spaces."""

    prev_space = _embed_in_basic_space(prev_pcs, key)
    chord_space = _embed_in_basic_space(chord_pcs, key)
    return len(prev_space ^ chord_space)


def chord_tone_inclusion(chosen_chord: Any, melody_event: tuple[int, int, int] | None, prev_chord: Any, key: str | None) -> float:
    """Reward melody pitch-class inclusion in the selected chord tones."""

    chosen = _coerce_chord(chosen_chord)
    melody_pc = _melody_pc(melody_event)
    if chosen["is_rest"] or melody_pc is None:
        return 0.0
    return 1.0 if melody_pc in set(chosen["pcs"]) else 0.0


def key_fit_proxy(chosen_chord: Any, melody_event: tuple[int, int, int] | None, prev_chord: Any, key: str | None) -> float:
    """Inference proxy for likely key-relative chord choices.

    The training code computes a supervised baseline against corpus ground
    truth. At inference time that target is unavailable, so this component
    scores simple key fit plus melody inclusion instead.
    """

    chosen = _coerce_chord(chosen_chord)
    if chosen["is_rest"]:
        return -1.0
    tonic_pc, mode = parse_key(key)
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    if not chosen["pcs"]:
        return 0.0
    diatonic_hits = sum(1 for pc in chosen["pcs"] if ((pc - tonic_pc) % 12) in scale)
    key_fit = diatonic_hits / len(chosen["pcs"])
    melody_fit = chord_tone_inclusion(chosen, melody_event, prev_chord, key)
    return round((0.75 * key_fit) + (0.25 * melody_fit), 6)


def compute_attribution(
    chosen: Any,
    melody_event: tuple[int, int, int] | None,
    prev_chord: Any,
    key: str | None,
) -> dict[str, float]:
    """Compute per-component reward attribution for one selected chord.

    TODO: wire ``mutual_info`` once the Mutual_Chord checkpoints are loaded in
    the inference path.
    """

    return {
        "harmony_rule": harmony_rule(chosen, melody_event, prev_chord, key),
        "progression_penalty": progression_penalty(chosen, melody_event, prev_chord, key),
        "chord_tone_inclusion": chord_tone_inclusion(chosen, melody_event, prev_chord, key),
        "mutual_info": 0.0,
        "key_fit_proxy": key_fit_proxy(chosen, melody_event, prev_chord, key),
    }


def _coerce_chord(chord: Any) -> dict[str, Any] | None:
    if chord is None:
        return None
    if hasattr(chord, "model_dump"):
        chord = chord.model_dump()
    if not isinstance(chord, dict):
        chord = vars(chord)
    return {
        "is_rest": bool(chord.get("is_rest", False)),
        "octave": chord.get("octave"),
        "inversion": chord.get("inversion"),
        "pcs": [int(pc) % 12 for pc in chord.get("pcs", [])],
        "pc_names": list(chord.get("pc_names", [])),
    }


def _embed_in_basic_space(pcs: set[int], key: str | None) -> set[tuple[str, int]]:
    normalized = {int(pc) % 12 for pc in pcs}
    tonic_pc, mode = parse_key(key)
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    root_pc = _infer_root_pc(normalized)
    space: set[tuple[str, int]] = {
        ("chromatic", pc)
        for pc in range(12)
    }
    space.update(("scale", (tonic_pc + pc) % 12) for pc in scale)
    space.update(("chord", pc) for pc in normalized)
    if root_pc is not None:
        space.add(("root", root_pc))
        space.add(("fifth", (root_pc + 7) % 12))
    return space


def _infer_root_pc(pcs: set[int]) -> int | None:
    if not pcs:
        return None
    for template in QUALITY_TEMPLATES:
        for root_pc in range(12):
            template_pcs = {(root_pc + interval) % 12 for interval in template.intervals}
            if template_pcs == pcs:
                return root_pc
    return min(pcs)


def _melody_pc(melody_event: tuple[int, int, int] | None) -> int | None:
    if melody_event is None:
        return None
    pitch = int(melody_event[0])
    return None if pitch == 0 else pitch % 12


def _harmonic_level(note1: int, note2: int) -> int:
    pd = abs((note1 - note2) % 12)
    pd = min(pd, 12 - pd)
    if pd == 0:
        return 10
    if pd in {5, 7}:
        return 8
    if pd in {4, 8}:
        return 6
    if pd in {3, 9}:
        return 5
    if pd in {2, 10}:
        return 3
    if pd in {1, 11}:
        return 1
    return 0


def _same_chord(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["is_rest"] == right["is_rest"]
        and left.get("octave") == right.get("octave")
        and left.get("inversion") == right.get("inversion")
        and sorted(left["pcs"]) == sorted(right["pcs"])
    )
