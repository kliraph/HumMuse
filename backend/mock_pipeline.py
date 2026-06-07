"""Deterministic GPT-down fallbacks for the session-centric API.

Everything in this module is invoked only when the GPT pipeline is
unavailable (no credentials, transport error, schema validation failure).
Live production helpers - `add_history`, `append_chat_message`,
`now_iso` - moved to `backend/history.py`.

Functions kept here:

* :func:`build_progressions`        - wrap chord-symbol lists into
                                      :class:`Progression` objects with
                                      placeholder scores. Used by
                                      ``/melody/from-hum`` and ``/chords/from-lyrics``
                                      when there is no melody to drive the
                                      real DQN.
* :func:`build_emotion_vector`      - 5-preset mood string ->
                                      :class:`EmotionVector`. Currently the
                                      only mood-to-vector adapter; live in
                                      production until a real classifier
                                      replaces it (planned).
* :func:`build_lyric_suggestions`   - template-string interpolation;
                                      fallback for
                                      ``lyric_responder.generate_lyric_suggestions``.
* :func:`build_chat_reply`          - explanation_report.summary +
                                      chord-detail concatenation; fallback
                                      for ``chat_responder.build_chat_reply``.
* :func:`build_refinement_plan`     - bilingual (EN+RU) keyword heuristic;
                                      fallback for
                                      ``refinement_parser.parse_refinement_instruction``.
"""

from __future__ import annotations

from typing import Iterable

from ml.gpt.syllable import count_syllables
from shared.schemas import (
    ChatMessage,
    LyricSuggestion,
    Progression,
    RefinementOp,
    RefinementPlan,
    SessionState,
)
from backend.history import now_iso

# The canonical mood/emotion taxonomy lives in shared.schemas so the backend
# and ml layers share one definition. Re-exported here for backward
# compatibility — existing callers (mood_responder, refinement_executor) import
# these names from backend.mock_pipeline.
from shared.schemas import (  # noqa: F401  (re-exported)
    EMOTION_PRESET_LABELS,
    _EMOTION_PRESETS,
    build_emotion_vector,
    lookup_emotion_preset,
    normalize_mood_label,
)


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
        timestamp=now_iso(),
    )
