"""Tests for GPT lyric syllable validation and retry helpers."""

from __future__ import annotations

import pytest

from ml.gpt.syllable import (
    build_syllable_retry_instruction,
    count_syllables,
    extract_lyric_lines,
    generate_with_syllable_retries,
    validate_syllable_targets,
)


def test_count_syllables_counts_words_and_common_endings() -> None:
    assert count_syllables("beneath") == 2
    assert count_syllables("Hold the line") == 3
    assert count_syllables("Under city lights") == 5


def test_validate_syllable_targets_accepts_exact_counts() -> None:
    validation = validate_syllable_targets(
        ["Hold the line", "Under city lights"],
        [3, 5],
    )

    assert validation.valid is True
    assert validation.actual_counts == [3, 5]
    assert validation.missing_line_count == 0
    assert validation.extra_line_count == 0


def test_validate_syllable_targets_reports_line_deltas_and_shape_errors() -> None:
    validation = validate_syllable_targets(
        ["Hold the bright line", "Under city lights", "extra line"],
        [3, 5],
    )

    assert validation.valid is False
    assert validation.line_results[0].target == 3
    assert validation.line_results[0].actual == 4
    assert validation.line_results[0].delta == 1
    assert validation.extra_line_count == 1


def test_validate_syllable_targets_honors_tolerance() -> None:
    validation = validate_syllable_targets(["Hold the bright line"], [3], tolerance=1)

    assert validation.valid is True


def test_extract_lyric_lines_accepts_common_response_shapes() -> None:
    assert extract_lyric_lines({"lyrics": ["Hold the line", "Under city lights"]}) == [
        "Hold the line",
        "Under city lights",
    ]
    assert extract_lyric_lines({"text": "Hold the line\nUnder city lights"}) == [
        "Hold the line",
        "Under city lights",
    ]


def test_retry_instruction_names_failed_lines() -> None:
    validation = validate_syllable_targets(["Hold the bright line"], [3])
    instruction = build_syllable_retry_instruction(validation)

    assert "line 1" in instruction
    assert "target 3" in instruction
    assert "actual 4" in instruction


def test_generate_with_syllable_retries_stops_when_valid() -> None:
    calls: list[list[dict[str, str]]] = []
    responses = [
        {"lyrics": ["Hold the bright line", "Under city lights"]},
        {"lyrics": ["Hold the line", "Under city lights"]},
    ]

    def generator(messages):
        calls.append(messages)
        return responses.pop(0)

    result = generate_with_syllable_retries(
        generator,
        [{"role": "user", "content": "draft"}],
        [3, 5],
    )

    assert result.validation.valid is True
    assert result.lyrics == ["Hold the line", "Under city lights"]
    assert result.attempts == 2
    assert len(calls) == 2
    assert "line 1" in calls[1][-1]["content"]


def test_generate_with_syllable_retries_uses_at_most_two_retries_by_default() -> None:
    calls = 0

    def generator(_messages):
        nonlocal calls
        calls += 1
        return {"lyrics": ["Hold the bright line"]}

    result = generate_with_syllable_retries(
        generator,
        [{"role": "user", "content": "draft"}],
        [3],
    )

    assert result.validation.valid is False
    assert result.attempts == 3
    assert calls == 3


def test_generate_with_syllable_retries_validates_retry_configuration() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        generate_with_syllable_retries(lambda _messages: [], [], [3], max_retries=-1)

    with pytest.raises(ValueError, match="tolerance"):
        validate_syllable_targets(["line"], [1], tolerance=-1)
