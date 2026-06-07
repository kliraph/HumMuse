"""Syllable counting, lyric validation, and retry helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-zа-яё]+(?:'[a-zа-яё]+)?", re.IGNORECASE)
_ENGLISH_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_RUSSIAN_VOWEL_RE = re.compile(r"[аеёиоуыэюя]", re.IGNORECASE)
_CYRILLIC_LETTER_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN_LETTER_RE = re.compile(r"[a-z]", re.IGNORECASE)
# A plural/3rd-person "-es" is its own syllable (/ɪz/) after a sibilant stem:
# s, z, x, soft c/g, or the digraphs ch/sh. So "dishes", "races", "changes"
# must NOT have the "-es" decrement applied. (Rare hard-/k/ "aches" is mis-served
# by the soft-c/g shortcut, but that's an acceptable miss for a heuristic.)
_SIBILANT_PLURAL_RE = re.compile(r"(?:[szxcg]|[cs]h)es$")
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
    """Count approximate syllables in free text (English and Russian)."""
    return sum(count_word_syllables(word) for word in _WORD_RE.findall(text))


def count_word_syllables(word: str) -> int:
    """Count approximate syllables in a single word.

    Dispatches per script: Russian words count one syllable per vowel letter
    (а, е, ё, и, о, у, ы, э, ю, я); English words use a vowel-group heuristic
    with silent-e / -es / -ed adjustments. Mixed-script tokens (rare; usually
    transliterations) sum both counts so the line total isn't undercounted.
    """
    lowered = word.lower()
    has_cyrillic = bool(_CYRILLIC_LETTER_RE.search(lowered))
    has_latin = bool(_LATIN_LETTER_RE.search(lowered))
    if has_cyrillic and not has_latin:
        return _count_russian_word_syllables(lowered)
    if has_latin and not has_cyrillic:
        return _count_english_word_syllables(lowered)
    if has_cyrillic and has_latin:
        return _count_russian_word_syllables(lowered) + _count_english_word_syllables(lowered)
    return 0


def _count_russian_word_syllables(word: str) -> int:
    return len(_RUSSIAN_VOWEL_RE.findall(word))


def _count_english_word_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word)
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    syllables = len(_ENGLISH_VOWEL_GROUP_RE.findall(cleaned))
    # Silent terminal "e" ("make", "tone"). The vowel-group regex already
    # collapses pronounced vowel-vowel endings ("agree", "value", "argue") into
    # one group with the preceding vowel, so guarding only "le"/"ye" would
    # over-strip those — require the char before the final "e" to be a
    # consonant so we only drop a genuinely silent, standalone "e".
    if (
        cleaned.endswith("e")
        and not cleaned.endswith(("le", "ye"))
        and len(cleaned) >= 2
        and cleaned[-2] not in "aeiou"
        and syllables > 1
    ):
        syllables -= 1
    # Silent "-es"/"-ed": both add a syllable when the stem already ends in the
    # matching sibilant ("-es" after s/z/x/soft-c-g/ch/sh) or stop ("-ed" after
    # t/d); otherwise they are silent and the vowel group must be removed.
    if (
        cleaned.endswith("es")
        and not _SIBILANT_PLURAL_RE.search(cleaned)
        and syllables > 1
    ):
        syllables -= 1
    elif cleaned.endswith("ed") and not cleaned.endswith(("ted", "ded")) and syllables > 1:
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
