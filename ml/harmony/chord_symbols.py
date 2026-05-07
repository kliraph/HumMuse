"""Deterministic chord-symbol derivation for native chord outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.schemas import ChordDistribution, ChordSymbolDerivation

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC = {name: index for index, name in enumerate(PC_NAMES)}
NAME_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


@dataclass(frozen=True)
class QualityTemplate:
    quality: str
    intervals: tuple[int, ...]
    suffix: str
    roman_suffix: str


QUALITY_TEMPLATES: tuple[QualityTemplate, ...] = (
    QualityTemplate("major", (0, 4, 7), "", ""),
    QualityTemplate("minor", (0, 3, 7), "m", ""),
    QualityTemplate("dim", (0, 3, 6), "dim", "dim"),
    QualityTemplate("aug", (0, 4, 8), "aug", "+"),
    QualityTemplate("sus2", (0, 2, 7), "sus2", "sus2"),
    QualityTemplate("sus4", (0, 5, 7), "sus4", "sus4"),
    QualityTemplate("7", (0, 4, 7, 10), "7", "7"),
    QualityTemplate("m7", (0, 3, 7, 10), "m7", "7"),
    QualityTemplate("maj7", (0, 4, 7, 11), "maj7", "maj7"),
    QualityTemplate("dim7", (0, 3, 6, 9), "dim7", "dim7"),
    QualityTemplate("half-dim7", (0, 3, 6, 10), "m7b5", "m7b5"),
    QualityTemplate("6", (0, 4, 7, 9), "6", "6"),
    QualityTemplate("m6", (0, 3, 7, 9), "m6", "6"),
)

COMMON_KEY_QUALITIES = {
    "major": 5,
    "minor": 5,
    "7": 4,
    "m7": 4,
    "dim": 3,
    "half-dim7": 3,
    "maj7": 2,
    "6": 2,
    "m6": 1,
    "sus2": 1,
    "sus4": 1,
    "aug": 0,
    "dim7": 0,
}

MAJOR_ROMANS = {
    0: ("I", "tonic"),
    2: ("ii", "predominant"),
    4: ("iii", "tonic"),
    5: ("IV", "subdominant"),
    7: ("V", "dominant"),
    9: ("vi", "tonic"),
    11: ("vii", "dominant"),
}
MINOR_ROMANS = {
    0: ("i", "tonic"),
    2: ("ii", "predominant"),
    3: ("III", "tonic"),
    5: ("iv", "subdominant"),
    7: ("V", "dominant"),
    8: ("VI", "subdominant"),
    10: ("VII", "dominant"),
}


@dataclass(frozen=True)
class Candidate:
    root_pc: int
    template: QualityTemplate
    bass_pc: int
    symbol: str
    roman_numeral: str
    harmonic_function: str
    score: float


def derive_chord_symbol(
    native_distribution: ChordDistribution | dict[str, Any],
    key: str | None = "C major",
    *,
    previous_symbol: str | None = None,
    previous_pcs: list[int] | None = None,
    next_symbol: str | None = None,
    next_pcs: list[int] | None = None,
) -> ChordSymbolDerivation:
    """Derive a deterministic chord-symbol reading from one native model output.

    The first pass intentionally resolves isolated ambiguities with a simple
    ranking: bass/root evidence from the inversion head, then diatonic fit, then
    common-practice quality preference. Some ties need harmonic context, so the
    returned metadata records the competing symbols as a failure mode.
    """

    distribution = _coerce_distribution(native_distribution)
    if previous_symbol is None:
        previous_symbol = distribution.context.prev_chord_symbol
    if previous_pcs is None:
        previous_pcs = distribution.context.prev_chord_pcs
    selected = distribution.selected
    if selected.get("is_rest"):
        return ChordSymbolDerivation(
            symbol="N.C.",
            root="N.C.",
            root_pc=0,
            quality="rest",
            bass="N.C.",
            bass_pc=0,
            pitch_classes=[],
            inversion=0,
            roman_numeral="N.C.",
            harmonic_function="chromatic",
            confidence=distribution.rest.get("rest", 1.0),
            candidate_symbols=[],
            failure_modes=[],
        )

    pitch_classes = sorted({int(pc) % 12 for pc in selected.get("pcs", [])})
    inversion = int(selected.get("inversion", _argmax_label(distribution.inversion)))
    if not pitch_classes:
        pitch_classes = _top_pitch_classes(distribution)
    observed_bass_pc = _observed_bass_from_inversion(pitch_classes, inversion)

    tonic_pc, mode = parse_key(key)
    candidates = _candidate_readings(pitch_classes, inversion, observed_bass_pc, tonic_pc, mode)
    if not candidates:
        fallback_root = observed_bass_pc if observed_bass_pc in pitch_classes else pitch_classes[0]
        symbol = _pc_name(fallback_root)
        return ChordSymbolDerivation(
            symbol=symbol,
            root=_pc_name(fallback_root),
            root_pc=fallback_root,
            quality="unknown",
            bass=_pc_name(observed_bass_pc),
            bass_pc=observed_bass_pc,
            pitch_classes=pitch_classes,
            inversion=inversion,
            roman_numeral="?",
            harmonic_function="chromatic",
            confidence=_membership_confidence(distribution, pitch_classes),
            candidate_symbols=[],
            failure_modes=["no supported quality matched the selected pitch-class set"],
        )

    top_score = candidates[0].score
    tied = [candidate for candidate in candidates if abs(candidate.score - top_score) < 1e-9]
    chosen = (
        _break_contextual_tie(tied, previous_symbol, previous_pcs, next_symbol, next_pcs)
        if len(tied) > 1
        else candidates[0]
    )
    candidate_symbols = [_symbol(candidate.root_pc, candidate.template, candidate.root_pc) for candidate in candidates]
    failure_modes = []
    if len(candidate_symbols) > 1:
        failure_modes.append(
            "ambiguous pitch-class set; first pass used inversion-derived bass, key fit, and common-quality preference"
        )

    return ChordSymbolDerivation(
        symbol=chosen.symbol,
        root=_pc_name(chosen.root_pc),
        root_pc=chosen.root_pc,
        quality=chosen.template.quality,
        bass=_pc_name(chosen.bass_pc),
        bass_pc=chosen.bass_pc,
        pitch_classes=pitch_classes,
        inversion=inversion,
        roman_numeral=chosen.roman_numeral,
        harmonic_function=chosen.harmonic_function,
        confidence=_membership_confidence(distribution, pitch_classes),
        candidate_symbols=candidate_symbols,
        failure_modes=failure_modes,
    )


def parse_key(key: str | None) -> tuple[int, str]:
    if not key:
        return 0, "major"
    parts = key.strip().split()
    tonic = parts[0] if parts else "C"
    mode = parts[1].lower() if len(parts) > 1 else "major"
    return NAME_TO_PC.get(tonic, 0), "minor" if mode.startswith("min") else "major"


def _candidate_readings(
    pitch_classes: list[int],
    inversion: int,
    observed_bass_pc: int,
    tonic_pc: int,
    mode: str,
) -> list[Candidate]:
    pitch_set = set(pitch_classes)
    candidates = []
    for root_pc in pitch_classes:
        intervals = tuple(sorted((pc - root_pc) % 12 for pc in pitch_classes))
        for template in QUALITY_TEMPLATES:
            if intervals != tuple(sorted(template.intervals)):
                continue
            bass_pc = _candidate_bass(root_pc, template.intervals, inversion)
            roman_numeral, harmonic_function = _roman_and_function(root_pc, template, tonic_pc, mode)
            score = _candidate_score(
                root_pc=root_pc,
                template=template,
                bass_pc=bass_pc,
                observed_bass_pc=observed_bass_pc,
                harmonic_function=harmonic_function,
                pitch_set=pitch_set,
            )
            candidates.append(
                Candidate(
                    root_pc=root_pc,
                    template=template,
                    bass_pc=observed_bass_pc,
                    symbol=_symbol(root_pc, template, observed_bass_pc),
                    roman_numeral=roman_numeral,
                    harmonic_function=harmonic_function,
                    score=score,
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _candidate_score(
    *,
    root_pc: int,
    template: QualityTemplate,
    bass_pc: int,
    observed_bass_pc: int,
    harmonic_function: str,
    pitch_set: set[int],
) -> float:
    score = 0.0
    if root_pc == observed_bass_pc:
        score += 5.0
    if bass_pc == observed_bass_pc:
        score += 3.0
    if harmonic_function != "chromatic":
        score += 2.0
    score += COMMON_KEY_QUALITIES.get(template.quality, 0) / 10.0
    if template.quality in {"6", "m6"} and len(pitch_set) == 4:
        score -= 0.1
    return score


def _roman_and_function(
    root_pc: int,
    template: QualityTemplate,
    tonic_pc: int,
    mode: str,
) -> tuple[str, str]:
    degree = (root_pc - tonic_pc) % 12
    roman_map = MINOR_ROMANS if mode == "minor" else MAJOR_ROMANS
    if degree not in roman_map:
        return "?", "chromatic"
    base_roman, harmonic_function = roman_map[degree]
    roman = _quality_adjusted_roman(base_roman, template)
    return f"{roman}{template.roman_suffix}", harmonic_function


def _quality_adjusted_roman(base_roman: str, template: QualityTemplate) -> str:
    if template.quality in {"minor", "m7", "m6", "dim", "dim7", "half-dim7"}:
        return base_roman.lower()
    if template.quality in {"major", "7", "maj7", "6", "sus2", "sus4", "aug"}:
        return base_roman.upper()
    return base_roman


def _symbol(root_pc: int, template: QualityTemplate, bass_pc: int) -> str:
    base = f"{_pc_name(root_pc)}{template.suffix}"
    if bass_pc != root_pc:
        return f"{base}/{_pc_name(bass_pc)}"
    return base


def _candidate_bass(root_pc: int, intervals: tuple[int, ...], inversion: int) -> int:
    tones = [(root_pc + interval) % 12 for interval in intervals]
    if len(tones) == 3:
        index_by_inversion = {0: 0, 2: 1, 1: 2}
    else:
        index_by_inversion = {0: 0, 3: 1, 2: 2, 1: 3}
    return tones[index_by_inversion.get(inversion, 0)]


def _observed_bass_from_inversion(pitch_classes: list[int], inversion: int) -> int:
    if not pitch_classes:
        return 0
    sorted_pcs = sorted(pitch_classes)
    if len(sorted_pcs) == 3:
        index_by_inversion = {0: 0, 2: 1, 1: 2}
    else:
        index_by_inversion = {0: 0, 3: 1, 2: 2, 1: 3}
    return sorted_pcs[index_by_inversion.get(inversion, 0) % len(sorted_pcs)]


def _membership_confidence(distribution: ChordDistribution, pitch_classes: list[int]) -> float:
    if not pitch_classes:
        return distribution.rest.get("rest", 0.0)
    values = [distribution.pitch_class_membership[pc] for pc in pitch_classes]
    return round(sum(values) / len(values), 6)


def _top_pitch_classes(distribution: ChordDistribution) -> list[int]:
    memberships = distribution.pitch_class_membership[:12]
    ranked = sorted(range(12), key=lambda pc: memberships[pc], reverse=True)
    triad_sentinel = distribution.pitch_class_membership[12]
    fourth_value = memberships[ranked[3]]
    chord_size = 3 if triad_sentinel > fourth_value else 4
    return sorted(ranked[:chord_size])


def _argmax_label(values: dict[str, float]) -> int:
    if not values:
        return 0
    return int(max(values, key=values.get))


def _coerce_distribution(native_distribution: ChordDistribution | dict[str, Any]) -> ChordDistribution:
    if isinstance(native_distribution, ChordDistribution):
        return native_distribution
    return ChordDistribution.model_validate(native_distribution)


def _pc_name(pc: int) -> str:
    return PC_NAMES[pc % 12]


def _break_contextual_tie(
    candidates: list[Candidate],
    previous_symbol: str | None,
    previous_pcs: list[int] | None,
    next_symbol: str | None,
    next_pcs: list[int] | None,
) -> Candidate:
    context_sets = [
        {int(pc) % 12 for pc in pcs}
        for pcs in (previous_pcs, next_pcs)
        if pcs
    ]
    if context_sets:
        return max(
            candidates,
            key=lambda candidate: (
                sum(len(_candidate_pitch_classes(candidate) & context) for context in context_sets),
                -candidates.index(candidate),
            ),
        )

    if not previous_symbol and not next_symbol:
        return candidates[0]
    context = f"{previous_symbol or ''} {next_symbol or ''}"
    for candidate in candidates:
        if _pc_name(candidate.root_pc) in context:
            return candidate
    return candidates[0]


def _candidate_pitch_classes(candidate: Candidate) -> set[int]:
    return {
        (candidate.root_pc + interval) % 12
        for interval in candidate.template.intervals
    }



