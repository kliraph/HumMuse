"""Syllable counting, lyric validation, and retry helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class SyllableLineResult:
    """Validation details for one lyric line."""

    line_index: int
    text: str
    target: int
    actual: int
    delta: int
    valid: bool


@dataclass(frozen=True)
class SyllableValidationResult:
    """Aggregate syllable validation result."""

    valid: bool
    line_results: list[SyllableLineResult]
    missing_line_count: int = 0
    extra_line_count: int = 0

    @property
    def actual_counts(self) -> list[int]:
        return [result.actual for result in self.line_results]


@dataclass(frozen=True)
class LyricRetryResult:
    """Result from retrying lyric generation until syllable targets match."""

    lyrics: list[str]
    validation: SyllableValidationResult
    attempts: int
    raw_response: Any


def count_syllables(text: str) -> int:
    """Count approximate English syllables in free text."""
    return sum(count_word_syllables(word) for word in _WORD_RE.findall(text))


def count_word_syllables(word: str) -> int:
    """Count approximate English syllables in a single word."""
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    syllables = len(_VOWEL_GROUP_RE.findall(cleaned))
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and syllables > 1:
        syllables -= 1
    if cleaned.endswith(("es", "ed")) and not cleaned.endswith(("ted", "ded")) and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def validate_syllable_targets(
    lyrics: str | Sequence[str],
    syllable_targets_per_line: Sequence[int],
    *,
    tolerance: int = 0,
) -> SyllableValidationResult:
    """Validate each lyric line against its target syllable count."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    targets = [int(target) for target in syllable_targets_per_line]
    if any(target < 0 for target in targets):
        raise ValueError("syllable targets must be >= 0")

    lines = extract_lyric_lines(lyrics)
    line_results = [
        _validate_line(index, line, targets[index], tolerance=tolerance)
        for index, line in enumerate(lines[: len(targets)])
    ]
    missing_line_count = max(0, len(targets) - len(lines))
    extra_line_count = max(0, len(lines) - len(targets))
    valid = (
        missing_line_count == 0
        and extra_line_count == 0
        and all(result.valid for result in line_results)
    )
    return SyllableValidationResult(
        valid=valid,
        line_results=line_results,
        missing_line_count=missing_line_count,
        extra_line_count=extra_line_count,
    )


def generate_with_syllable_retries(
    generator: Callable[[list[dict[str, str]]], Any],
    messages: Sequence[Mapping[str, str]],
    syllable_targets_per_line: Sequence[int],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tolerance: int = 0,
) -> LyricRetryResult:
    """
    Call a lyric generator until syllable targets validate or retries are spent.

    `generator` receives OpenAI-style messages and may return a dict with a
    `lyrics` field, a list of lyric lines, or a newline-delimited string.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    current_messages = [dict(message) for message in messages]
    last_response: Any = None
    last_lyrics: list[str] = []
    last_validation = validate_syllable_targets([], syllable_targets_per_line, tolerance=tolerance)

    for attempt in range(max_retries + 1):
        last_response = generator(current_messages)
        last_lyrics = extract_lyric_lines(last_response)
        last_validation = validate_syllable_targets(
            last_lyrics,
            syllable_targets_per_line,
            tolerance=tolerance,
        )
        if last_validation.valid:
            return LyricRetryResult(
                lyrics=last_lyrics,
                validation=last_validation,
                attempts=attempt + 1,
                raw_response=last_response,
            )
        if attempt < max_retries:
            current_messages.append(
                {
                    "role": "user",
                    "content": build_syllable_retry_instruction(last_validation),
                }
            )

    return LyricRetryResult(
        lyrics=last_lyrics,
        validation=last_validation,
        attempts=max_retries + 1,
        raw_response=last_response,
    )


def extract_lyric_lines(response: Any) -> list[str]:
    """Extract lyric lines from common GPT response shapes."""
    if isinstance(response, str):
        return _split_lyric_text(response)
    if isinstance(response, Mapping):
        lyrics = response.get("lyrics")
        if lyrics is not None:
            return extract_lyric_lines(lyrics)
        text = response.get("text")
        if text is not None:
            return extract_lyric_lines(text)
    if isinstance(response, Sequence) and not isinstance(response, (bytes, bytearray, str)):
        return [str(line).strip() for line in response if str(line).strip()]
    return []


def build_syllable_retry_instruction(validation: SyllableValidationResult) -> str:
    """Build a corrective prompt message from validation errors."""
    problems: list[str] = []
    for result in validation.line_results:
        if result.valid:
            continue
        direction = "too long" if result.delta > 0 else "too short"
        problems.append(
            f"line {result.line_index + 1} is {direction}: target {result.target}, actual {result.actual}"
        )
    if validation.missing_line_count:
        problems.append(f"missing {validation.missing_line_count} required line(s)")
    if validation.extra_line_count:
        problems.append(f"remove {validation.extra_line_count} extra line(s)")
    detail = "; ".join(problems) if problems else "syllable counts did not match"
    return (
        "Revise the lyric JSON so each line exactly matches the requested "
        f"syllable target. Problems: {detail}."
    )


def _validate_line(index: int, line: str, target: int, *, tolerance: int) -> SyllableLineResult:
    actual = count_syllables(line)
    delta = actual - target
    return SyllableLineResult(
        line_index=index,
        text=line,
        target=target,
        actual=actual,
        delta=delta,
        valid=abs(delta) <= tolerance,
    )


def _split_lyric_text(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
