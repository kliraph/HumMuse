"""Post-hoc deterministic chord theory annotations."""

from __future__ import annotations

from typing import Any

from ml.harmony.chord_symbols import PC_NAMES, derive_chord_symbol
from shared.schemas import ChordDistribution, ChordAnnotation, ChordSymbolDerivation

STRONG_BAR_POSITIONS = {0, 24}

ROMAN_PHRASES = {
    "I": "tonic - stable home base",
    "i": "tonic minor - stable home base in minor",
    "ii": "supertonic - predominant preparation",
    "iii": "mediant - tonic prolongation color",
    "IV": "subdominant - prepares motion away from tonic",
    "iv": "minor subdominant - darker predominant color",
    "V": "dominant - strong pull to tonic",
    "V7": "dominant seventh - strong pull to tonic",
    "vi": "submediant - tonic substitute, common in pop progressions",
    "vii": "leading-tone chord - unstable dominant function",
    "viidim": "leading-tone diminished chord - unstable dominant function",
}
FUNCTION_PHRASES = {
    "tonic": "tonic function - stable or substitute home area",
    "predominant": "predominant function - prepares dominant motion",
    "subdominant": "subdominant function - moves away from tonic",
    "dominant": "dominant function - points back toward tonic",
    "chromatic": "chromatic function - outside the basic diatonic table",
}


def annotate_chord_positions(
    native_distributions: list[ChordDistribution],
    melody_events: list[tuple[int, int, int]],
    *,
    key: str | None = "C major",
    derivations: list[ChordSymbolDerivation] | None = None,
) -> list[ChordAnnotation]:
    """Annotate each generated chord position with deterministic theory data."""

    resolved_derivations = derivations or [
        derive_chord_symbol(
            distribution,
            key=key,
            previous_symbol=distribution.context.prev_chord_symbol,
            previous_pcs=distribution.context.prev_chord_pcs,
        )
        for distribution in native_distributions
    ]
    annotations = []
    for position, derivation in enumerate(resolved_derivations):
        event = melody_events[position] if position < len(melody_events) else None
        distribution = native_distributions[position] if position < len(native_distributions) else None
        annotations.append(annotate_chord_position(position, derivation, event, distribution))
    return annotations


def annotate_chord_position(
    position: int,
    derivation: ChordSymbolDerivation,
    melody_event: tuple[int, int, int] | None,
    distribution: ChordDistribution | None = None,
) -> ChordAnnotation:
    strong_beat_notes = _strong_beat_notes(derivation.pitch_classes, melody_event)
    if strong_beat_notes:
        hits = sum(1 for note in strong_beat_notes if note["is_chord_tone"])
        alignment_percentage = hits / len(strong_beat_notes)
    else:
        alignment_percentage = 0.0

    return ChordAnnotation(
        position=position,
        symbol=derivation.symbol,
        root=derivation.root,
        quality=derivation.quality,
        bass=derivation.bass,
        roman_numeral=derivation.roman_numeral,
        function_label=derivation.harmonic_function,
        alignment_percentage=round(alignment_percentage, 6),
        q_margin=_annotation_q_margin(distribution),
        value_score=_annotation_value_score(distribution),
        strong_beat_notes=strong_beat_notes,
        template_phrase=template_phrase(derivation),
        derivation=derivation,
    )


def template_phrase(derivation: ChordSymbolDerivation) -> str:
    if derivation.roman_numeral in ROMAN_PHRASES:
        return ROMAN_PHRASES[derivation.roman_numeral]
    return FUNCTION_PHRASES.get(
        derivation.harmonic_function,
        "chromatic function - outside the basic diatonic table",
    )


def _strong_beat_notes(
    chord_pitch_classes: list[int],
    melody_event: tuple[int, int, int] | None,
) -> list[dict[str, Any]]:
    if melody_event is None:
        return []

    pitch, duration_type, bar_position = melody_event
    if pitch == 0 or bar_position not in STRONG_BAR_POSITIONS:
        return []

    pitch_class = pitch % 12
    chord_tones = set(chord_pitch_classes)
    return [
        {
            "pitch": pitch,
            "pitch_class": pitch_class,
            "pitch_class_name": PC_NAMES[pitch_class],
            "duration_type": duration_type,
            "bar_position": bar_position,
            "is_chord_tone": pitch_class in chord_tones,
        }
    ]


def _annotation_q_margin(distribution: ChordDistribution | None) -> float | None:
    if distribution is None:
        return None
    if distribution.selected.is_rest:
        return float(distribution.q_margin.rest)
    return min(
        float(distribution.q_margin.octave),
        float(distribution.q_margin.inversion),
        float(distribution.q_margin.pitch_class),
    )


def _annotation_value_score(distribution: ChordDistribution | None) -> float | None:
    if distribution is None:
        return None
    if distribution.selected.is_rest:
        return float(distribution.dueling.value_rest)
    return float(distribution.dueling.value_pc)

