"""Deterministic mock generators used by the early session-centric API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Iterable

from ml.melody_sketchpad.notes import melody_notes_to_events, pitch_to_midi
from shared.schemas import (
    Action,
    ChatMessage,
    EmotionVector,
    LyricSuggestion,
    MelodySuggestion,
    NoteEvent,
    Progression,
    RefinementOp,
    RefinementPlan,
    SessionState,
)
from ml.melody_sketchpad.profile import build_melody_profile

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def build_progressions(symbol_groups: Iterable[list[str]], mood: str) -> list[Progression]:
    progressions: list[Progression] = []
    base_score = 0.9
    harmonic_map = [
        "tonic extension",
        "predominant motion",
        "dominant release",
    ]
    for index, chords in enumerate(symbol_groups):
        progressions.append(
            Progression(
                chords=chords,
                score=max(0.5, base_score - (index * 0.1)),
                harmonic_function=harmonic_map[min(index, len(harmonic_map) - 1)],
                explanation=f"Mock {mood} progression with stable voice-leading and a clear cadence.",
            )
        )
    return progressions


def build_emotion_vector(mood: str) -> EmotionVector:
    presets = {
        "joyful": EmotionVector(valence=0.8, arousal=0.7),
        "uplift": EmotionVector(valence=0.7, arousal=0.6),
        "melancholic": EmotionVector(valence=-0.5, arousal=0.2),
        "tense": EmotionVector(valence=-0.3, arousal=0.8),
        "neutral": EmotionVector(valence=0.0, arousal=0.3),
    }
    return presets.get(mood, EmotionVector(valence=0.1, arousal=0.4))


def count_syllables(text: str) -> int:
    return sum(_count_word_syllables(word) for word in _WORD_RE.findall(text))


def _count_word_syllables(word: str) -> int:
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


def build_melody_suggestions(state: SessionState, count: int = 3) -> list[MelodySuggestion]:
    seed_notes = state.melody_notes or [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.9),
        NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=98, confidence=0.9),
        NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=100, confidence=0.9),
    ]
    last_onset = max(note.onset + note.duration for note in seed_notes)
    suggestions: list[MelodySuggestion] = []
    for index in range(count):
        offset = index + 1
        notes = [
            NoteEvent(
                pitch=min(127, note.pitch + offset),
                onset=last_onset + idx,
                duration=note.duration,
                velocity=max(1, min(127, note.velocity - index)),
                confidence=max(0.6, note.confidence - (index * 0.05)),
            )
            for idx, note in enumerate(seed_notes[-3:])
        ]
        suggestions.append(
            MelodySuggestion(
                midi_bytes=f"mock-midi-{state.session_id}-{index}".encode("utf-8"),
                notes=notes,
                explanation=f"Continuation {index + 1} mirrors the session contour with a {offset}-semitone lift.",
                coherence_score=max(0.55, 0.9 - (index * 0.12)),
            )
        )
    return suggestions


def build_lyric_suggestions(state: SessionState, mode: str, count: int = 3) -> list[LyricSuggestion]:
    seed = state.lyrics_text or "Hold the line until the chorus lands"
    templates = [
        f"{seed} beneath the city lights",
        f"{seed} while the skyline starts to sway",
        f"{seed} and let the morning answer back",
    ]
    return [
        LyricSuggestion(text=templates[index], mode=mode, syllable_count=count_syllables(templates[index]))
        for index in range(min(count, len(templates)))
    ]


def build_refinement_plan(instruction: str, target: str | None = None) -> RefinementPlan:
    lowered = instruction.lower()
    operations: list[RefinementOp] = []
    less_sad = any(phrase in lowered for phrase in ("less sad", "менее груст", "не так груст", "менее печал", "светлее", "повеселее"))

    if target == "chords" or any(
        word in lowered
        for word in (
            "jazz",
            "chord",
            "harmon",
            "sad",
            "dark",
            "myster",
            "джаз",
            "аккорд",
            "гармон",
            "груст",
            "печал",
            "темн",
            "тёмн",
            "мрач",
        )
    ):
        operations.append(
            RefinementOp(
                target="harmonizer",
                params={
                    "requested_target": target or "session",
                    "chord_complexity": "high" if "jazz" in lowered or "джаз" in lowered else "medium",
                    "emotion_valence": 0.2
                    if less_sad
                    else -0.2
                    if any(word in lowered for word in ("dark", "sad", "myster", "груст", "печал", "темн", "тёмн", "мрач"))
                    else 0.1,
                },
                rationale="Adjust harmonic color and chord-selection settings.",
            )
        )
    if target == "melody" or any(
        word in lowered
        for word in ("melody", "hook", "contour", "shorter", "short", "мелод", "хук", "короч", "припев", "куплет")
    ):
        operations.append(
            RefinementOp(
                target="melody_generator",
                params={
                    "requested_target": target or "session",
                    "contour": "rising" if "lift" in lowered or "up" in lowered else "varied",
                    "rhythmic_density": "lighter" if "space" in lowered else "steady",
                    "length": "shorter" if "short" in lowered or "короч" in lowered else "unchanged",
                },
                rationale="Adjust melodic shape or phrase length.",
            )
        )
    if target in ("suggestions", "lyrics") or any(
        word in lowered
        for word in ("lyric", "lyrics", "line", "option", "options", "write", "текст", "лирик", "строк", "вариант", "напиши", "напиш")
    ):
        option_count = 2 if "2" in lowered or "two" in lowered or "два" in lowered else 3
        operations.append(
            RefinementOp(
                target="lyric_generator",
                params={
                    "requested_target": target or "session",
                    "imagery": "more vivid",
                    "num_options": option_count,
                },
                rationale="Generate revised lyric options after the musical changes.",
            )
        )
    if not operations:
        operations.append(
            RefinementOp(
                target="session",
                params={"requested_target": target or "session"},
                rationale="Keep the request attached to the full session context.",
            )
        )

    interpretation = (
        f"Interpreted '{instruction}' as {len(operations)} ordered refinement operation(s) "
        f"while keeping the current session context."
    )
    return RefinementPlan(
        operations=operations,
        interpretation=interpretation,
    )


def build_chat_reply(state: SessionState, message: str) -> ChatMessage:
    summary = (
        state.explanation_report.summary
        if state.explanation_report is not None
        else "There is no explanation report yet, so this response stays at the session-summary level."
    )
    chord_detail = ""
    if state.chord_progressions:
        progression = state.chord_progressions[0]
        if progression.chord_annotations:
            annotation = progression.chord_annotations[0]
            chord_detail = (
                f" The first generated chord is {annotation.symbol} "
                f"({annotation.roman_numeral}, {annotation.function_label})."
            )
        elif progression.chords:
            chord_detail = f" The current top progression starts with {progression.chords[0]}."
    return ChatMessage(
        role="assistant",
        content=f"{summary} In response to '{message}', the stub explanation stays grounded in the current session data.{chord_detail}",
        timestamp=_now_iso(),
    )


def add_history(state: SessionState, kind: str, payload: dict[str, object], *, source: str = "api") -> SessionState:
    state.history.append(Action(kind=kind, payload=payload, source=source))
    return state


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def append_chat_message(state: SessionState, role: str, content: str) -> ChatMessage:
    message = ChatMessage(role=role, content=content, timestamp=_now_iso())
    state.chat_history.append(message)
    return message
