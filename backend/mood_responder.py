"""Infer mood from lyrics: GPT first, keyword heuristic fallback.

GPT path: ``pipeline.route_request("mood", state, lyrics_text)`` returns
``{valence, arousal, mood_label, rationale}`` in a single call. The
``mood_label`` is free-form and emitted in the same language as the
lyrics.

Fallback path: a deliberately small EN+RU keyword classifier that maps
to one of the curated mood presets in :func:`build_emotion_vector`. This
is the only mood inference that runs when GPT is unavailable; it's crude
but always available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from backend.logging_config import get_logger
from backend.mock_pipeline import build_emotion_vector
from ml.gpt.pipeline import GPTPipeline
from shared.schemas import EmotionVector, SessionState

LOGGER = get_logger("backend.mood_responder")

MoodSource = Literal["gpt", "fallback"]

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Each keyword bucket maps to a mood preset that exists in
# `build_emotion_vector`. The lists are intentionally small (target: the
# rough quadrants of a circumplex), not an exhaustive emotion lexicon —
# the fallback only fires when GPT is down.
#
# Ordering matters: the first bucket whose keywords match wins. Buckets
# are listed from most-specific to most-general.
_FALLBACK_BUCKETS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # (mood_label_en, mood_label_ru, preset_key, keywords)
    ("triumphant", "торжествующий", "joyful", (
        "triumph", "victory", "soar", "rise", "champion", "conquer",
        "победа", "торжеств", "вершин", "взлет", "взлёт",
    )),
    ("joyful", "радостный", "joyful", (
        "joy", "happy", "smile", "shine", "love", "sunshine", "dance",
        "радост", "счаст", "улыб", "люблю", "любовь", "солнц", "танц",
    )),
    ("hopeful", "обнадёживающий", "uplift", (
        "hope", "dawn", "rise", "light", "tomorrow", "promise",
        "надежд", "рассвет", "свет", "завтра", "обещан",
    )),
    ("anxious", "тревожный", "tense", (
        "anxious", "afraid", "fear", "worry", "panic", "racing", "trembl",
        "тревог", "страх", "боюсь", "беспокой", "пугаю", "паник",
    )),
    ("tense", "напряжённый", "tense", (
        "tense", "tight", "edge", "fight", "storm", "burn",
        "напряж", "буря", "бой", "гнев",
    )),
    ("melancholic", "меланхоличный", "melancholic", (
        "sad", "tear", "alone", "lonely", "lost", "fading", "rain",
        "груст", "печал", "слёз", "слез", "один", "одинок", "дожд", "тоск",
    )),
    ("reflective", "задумчивый", "neutral", (
        "remember", "memory", "still", "quiet", "wonder", "moonlight",
        "вспомина", "память", "тих", "задум", "лун",
    )),
    ("calm", "спокойный", "neutral", (
        "calm", "peace", "soft", "gentle", "breath",
        "спокой", "мир", "тих", "нежн", "дыха",
    )),
]


@dataclass(frozen=True)
class MoodResult:
    """Mood inference output: continuous vector + free-form label."""

    valence: float
    arousal: float
    mood_label: str
    rationale: str
    emotion_vector: EmotionVector
    source: MoodSource
    model_version: str | None
    cache_hit: bool
    latency_ms: float | None
    error: str | None


def infer_mood(
    lyrics_text: str,
    *,
    session_state: SessionState | None = None,
    language: str | None = None,
    pipeline: GPTPipeline | None = None,
) -> MoodResult:
    """Infer mood from lyrics, falling back to the keyword heuristic on failure."""
    text = (lyrics_text or "").strip()
    if not text:
        # No lyrics → return neutral. Don't bother the model.
        return _neutral_fallback("no_lyrics", language_for=text)

    if pipeline is None:
        return _keyword_fallback(text, language_for=text, reason="no_gpt_pipeline")

    stub_state = session_state or SessionState(session_id=uuid4(), lyrics_text=text)
    try:
        result = pipeline.route_request("mood", stub_state, text, language=language)
    except Exception as exc:
        LOGGER.info(
            "mood_infer_gpt_failed",
            error=str(exc),
            error_type=exc.__class__.__name__,
            lyrics_preview=text[:80],
        )
        return _keyword_fallback(text, language_for=text, reason=f"transport:{exc.__class__.__name__}")

    parsed = _coerce_parsed(result.parsed)
    if parsed is None:
        LOGGER.info(
            "mood_infer_empty_payload",
            payload_type=type(result.parsed).__name__,
            lyrics_preview=text[:80],
        )
        return _keyword_fallback(text, language_for=text, reason="empty_payload")

    valence = _clip_unit(parsed.get("valence"))
    arousal = _clip_unit(parsed.get("arousal"))
    mood_label = str(parsed.get("mood_label") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if not mood_label:
        LOGGER.info(
            "mood_infer_empty_label",
            lyrics_preview=text[:80],
        )
        return _keyword_fallback(text, language_for=text, reason="empty_label")

    LOGGER.info(
        "mood_infer_gpt_succeeded",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        valence=valence,
        arousal=arousal,
        mood_label_preview=mood_label[:80],
    )
    return MoodResult(
        valence=valence,
        arousal=arousal,
        mood_label=mood_label,
        rationale=rationale,
        emotion_vector=EmotionVector(valence=valence, arousal=arousal),
        source="gpt",
        model_version=result.model_version,
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        error=None,
    )


# --- fallback ----------------------------------------------------------------


def _keyword_fallback(text: str, *, language_for: str, reason: str) -> MoodResult:
    """Map lyrics to a curated mood preset via small EN+RU keyword lists."""
    lowered = text.lower()
    use_russian = _looks_russian(language_for)
    for label_en, label_ru, preset_key, keywords in _FALLBACK_BUCKETS:
        if _matches_any_keyword(lowered, keywords):
            label = label_ru if use_russian else label_en
            vector = build_emotion_vector(preset_key)
            return MoodResult(
                valence=vector.valence,
                arousal=vector.arousal,
                mood_label=label,
                rationale="",
                emotion_vector=vector,
                source="fallback",
                model_version=None,
                cache_hit=False,
                latency_ms=None,
                error=reason,
            )
    # Nothing matched → reflective default.
    label = "задумчивый" if use_russian else "reflective"
    vector = build_emotion_vector("neutral")
    return MoodResult(
        valence=vector.valence,
        arousal=vector.arousal,
        mood_label=label,
        rationale="",
        emotion_vector=vector,
        source="fallback",
        model_version=None,
        cache_hit=False,
        latency_ms=None,
        error=reason,
    )


def _neutral_fallback(reason: str, *, language_for: str) -> MoodResult:
    use_russian = _looks_russian(language_for)
    label = "нейтральный" if use_russian else "neutral"
    vector = build_emotion_vector("neutral")
    return MoodResult(
        valence=vector.valence,
        arousal=vector.arousal,
        mood_label=label,
        rationale="",
        emotion_vector=vector,
        source="fallback",
        model_version=None,
        cache_hit=False,
        latency_ms=None,
        error=reason,
    )


# --- helpers -----------------------------------------------------------------


def _coerce_parsed(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clip_unit(value: Any) -> float:
    """Clamp a numeric value to ``[-1, 1]``; non-numeric inputs become 0.0."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:  # NaN guard
        return 0.0
    return max(-1.0, min(1.0, numeric))


def _looks_russian(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def _matches_any_keyword(text_lower: str, keywords: tuple[str, ...]) -> bool:
    """Check if any keyword matches the text.

    Cyrillic stems use substring match (Russian morphology needs prefix
    matching for inflection — "груст" catches "грустно", "грустный",
    "грусть"). Latin-alphabet keywords use word-boundary regex so
    "light" doesn't accidentally match "moonlight" in the reflective
    bucket, and "rise" doesn't match arbitrary words like "surprise".
    """
    for keyword in keywords:
        if _CYRILLIC_RE.search(keyword):
            if keyword in text_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                return True
    return False
