"""Phase 4C-A.2.5 tests for deterministic chord-symbol derivation."""

from __future__ import annotations

import pytest

import ml.harmony.chord_symbols as chord_symbols
from ml.harmony.chord_symbols import derive_chord_symbol
from shared.schemas import ChordDistribution

PC_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel"]


def _distribution(pcs: list[int], inversion: int = 0) -> ChordDistribution:
    membership = [0.001] * 13
    for pc in pcs:
        membership[pc] = 0.9
    membership[12] = 0.95 if len(pcs) == 3 else 0.001
    total = sum(membership)
    membership = [value / total for value in membership]
    return ChordDistribution(
        position=0,
        rest={"rest": 0.01, "chord": 0.99},
        octave={"2": 0.15, "3": 0.85},
        inversion={str(index): 1.0 if index == inversion else 0.0 for index in range(4)},
        pitch_class={label: membership[index] for index, label in enumerate(PC_LABELS[:12])},
        pitch_class_membership=membership,
        pitch_class_labels=PC_LABELS,
        selected={
            "is_rest": False,
            "octave": 3,
            "inversion": inversion,
            "pcs": sorted(pcs),
            "pc_names": [PC_LABELS[pc] for pc in sorted(pcs)],
        },
    )


@pytest.mark.parametrize(
    ("pcs", "expected_symbol", "expected_quality"),
    [
        ([0, 4, 7], "C", "major"),
        ([0, 3, 7], "Cm", "minor"),
        ([0, 3, 6], "Cdim", "dim"),
        ([0, 4, 8], "Caug", "aug"),
        ([0, 2, 7], "Csus2", "sus2"),
        ([0, 5, 7], "Csus4", "sus4"),
        ([0, 4, 7, 10], "C7", "7"),
        ([0, 3, 7, 10], "Cm7", "m7"),
        ([0, 4, 7, 11], "Cmaj7", "maj7"),
        ([0, 3, 6, 9], "Cdim7", "dim7"),
        ([0, 3, 6, 10], "Cm7b5", "half-dim7"),
        ([0, 4, 7, 9], "C6", "6"),
        ([0, 3, 7, 9], "Cm6", "m6"),
    ],
)
def test_common_chord_types_round_trip_in_c(pcs: list[int], expected_symbol: str, expected_quality: str) -> None:
    derived = derive_chord_symbol(_distribution(pcs), key="C major")

    assert derived.symbol == expected_symbol
    assert derived.root == "C"
    assert derived.quality == expected_quality
    assert derived.bass == "C"
    assert derived.pitch_classes == sorted(pcs)


@pytest.mark.parametrize(
    ("pcs", "inversion", "expected_symbol", "expected_roman", "expected_function"),
    [
        ([0, 4, 7], 0, "C", "I", "tonic"),
        ([2, 5, 9], 0, "Dm", "ii", "predominant"),
        ([4, 7, 11], 0, "Em", "iii", "tonic"),
        ([5, 9, 0], 2, "F", "IV", "subdominant"),
        ([7, 11, 2], 2, "G", "V", "dominant"),
        ([9, 0, 4], 1, "Am", "vi", "tonic"),
        ([11, 2, 5], 1, "Bdim", "viidim", "dominant"),
        ([7, 11, 2, 5], 2, "G7", "V7", "dominant"),
    ],
)
def test_diatonic_c_major_roman_and_function_labels(
    pcs: list[int],
    inversion: int,
    expected_symbol: str,
    expected_roman: str,
    expected_function: str,
) -> None:
    derived = derive_chord_symbol(_distribution(pcs, inversion=inversion), key="C major")

    assert derived.symbol == expected_symbol
    assert derived.roman_numeral == expected_roman
    assert derived.harmonic_function == expected_function


def test_ambiguous_c6_am7_resolves_from_inversion_bass_implication() -> None:
    root_position = derive_chord_symbol(_distribution([0, 4, 7, 9], inversion=0), key="C major")
    a_bass = derive_chord_symbol(_distribution([0, 4, 7, 9], inversion=1), key="C major")

    assert root_position.symbol == "C6"
    assert root_position.bass == "C"
    assert a_bass.symbol == "Am7"
    assert a_bass.bass == "A"
    assert "C6" in a_bass.candidate_symbols
    assert "Am7" in a_bass.candidate_symbols
    assert a_bass.failure_modes


def test_ambiguous_d_f_a_c_prefers_dmin7_when_inversion_implies_d_bass() -> None:
    derived = derive_chord_symbol(_distribution([0, 2, 5, 9], inversion=3), key="C major")

    assert derived.symbol == "Dm7"
    assert derived.root == "D"
    assert derived.bass == "D"
    assert "F6" in derived.candidate_symbols
    assert derived.failure_modes


def test_contextual_tie_break_reads_previous_symbol_from_distribution_context(monkeypatch) -> None:
    distribution = _distribution([0, 4, 7])
    payload = distribution.model_dump()
    payload["context"]["prev_chord_symbol"] = "Am"
    distribution = ChordDistribution.model_validate(payload)
    tied_candidates = [
        chord_symbols.Candidate(
            root_pc=0,
            template=chord_symbols.QUALITY_TEMPLATES[0],
            bass_pc=0,
            symbol="C",
            roman_numeral="I",
            harmonic_function="tonic",
            score=1.0,
        ),
        chord_symbols.Candidate(
            root_pc=9,
            template=chord_symbols.QUALITY_TEMPLATES[1],
            bass_pc=9,
            symbol="Am",
            roman_numeral="vi",
            harmonic_function="tonic",
            score=1.0,
        ),
    ]
    monkeypatch.setattr(chord_symbols, "_candidate_readings", lambda *args: tied_candidates)

    derived = derive_chord_symbol(distribution, key="C major")

    assert derived.symbol == "Am"

