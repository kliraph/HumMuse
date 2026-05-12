"""Session context assembly for GPT prompt use cases."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ml.gpt.abc_serializer import session_state_to_abc
from ml.gpt.tier1_formatter import format_tier1_summary
from shared.schemas import ChatMessage, ChordProgression, SessionState

GPTUseCase = Literal["lyric", "refine", "explain"]

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_DEFAULT_RECENT_CHAT_MESSAGES = 6


@dataclass(frozen=True)
class SessionContext:
    """Structured and prompt-ready session context for a GPT use case."""

    use_case: GPTUseCase
    fields: dict[str, Any]

    def to_prompt_text(self) -> str:
        """Render context as stable, readable JSON for prompt templates."""
        return json.dumps(self.fields, ensure_ascii=False, indent=2, sort_keys=True)

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]


class SessionContextBuilder:
    """Build the exact session fields needed by each GPT use case."""

    def __init__(self, *, recent_chat_messages: int = _DEFAULT_RECENT_CHAT_MESSAGES) -> None:
        if recent_chat_messages < 0:
            raise ValueError("recent_chat_messages must be >= 0")
        self.recent_chat_messages = recent_chat_messages

    def build(
        self,
        use_case: GPTUseCase,
        session_state: SessionState,
        *,
        syllable_targets_per_line: Sequence[int] | None = None,
        last_user_question: str | None = None,
    ) -> SessionContext:
        if use_case == "lyric":
            return self.lyric_context(session_state, syllable_targets_per_line=syllable_targets_per_line)
        if use_case == "refine":
            return self.refine_context(session_state, syllable_targets_per_line=syllable_targets_per_line)
        if use_case == "explain":
            return self.explain_context(session_state, last_user_question=last_user_question)
        raise ValueError("use_case must be 'lyric', 'refine', or 'explain'")

    def lyric_context(
        self,
        session_state: SessionState,
        *,
        syllable_targets_per_line: Sequence[int] | None = None,
    ) -> SessionContext:
        return SessionContext(
            use_case="lyric",
            fields=self._lyric_base_fields(session_state, syllable_targets_per_line=syllable_targets_per_line),
        )

    def refine_context(
        self,
        session_state: SessionState,
        *,
        syllable_targets_per_line: Sequence[int] | None = None,
    ) -> SessionContext:
        fields = self._lyric_base_fields(session_state, syllable_targets_per_line=syllable_targets_per_line)
        fields["recent_chat_history"] = _chat_messages_to_dicts(
            _last_items(session_state.chat_history, self.recent_chat_messages)
        )
        fields["last_refinement_plan_results"] = _last_refinement_results(session_state.user_params)
        return SessionContext(use_case="refine", fields=fields)

    def explain_context(
        self,
        session_state: SessionState,
        *,
        last_user_question: str | None = None,
    ) -> SessionContext:
        return SessionContext(
            use_case="explain",
            fields={
                "session_abc": session_state_to_abc(session_state),
                "tier1_natural_language": format_tier1_summary(session_state.explanation_report),
                "explanation_report_tier_1": _model_to_dict(session_state.explanation_report),
                "chat_history": _chat_messages_to_dicts(session_state.chat_history),
                "last_user_question": last_user_question or _infer_last_user_question(session_state.chat_history),
            },
        )

    def _lyric_base_fields(
        self,
        session_state: SessionState,
        *,
        syllable_targets_per_line: Sequence[int] | None,
    ) -> dict[str, Any]:
        previous_lyrics = session_state.lyrics_text or ""
        return {
            "melody_profile": _model_to_dict(session_state.melody_profile),
            "chord_progressions": [_progression_summary(progression) for progression in session_state.chord_progressions],
            "syllable_targets_per_line": list(syllable_targets_per_line)
            if syllable_targets_per_line is not None
            else _derive_syllable_targets(previous_lyrics),
            "previous_lyrics": previous_lyrics,
            "emotion_vector": _model_to_dict(session_state.emotion_vector),
        }


def build_session_context(
    use_case: GPTUseCase,
    session_state: SessionState,
    *,
    syllable_targets_per_line: Sequence[int] | None = None,
    last_user_question: str | None = None,
    recent_chat_messages: int = _DEFAULT_RECENT_CHAT_MESSAGES,
) -> SessionContext:
    """Convenience wrapper around `SessionContextBuilder`."""
    return SessionContextBuilder(recent_chat_messages=recent_chat_messages).build(
        use_case,
        session_state,
        syllable_targets_per_line=syllable_targets_per_line,
        last_user_question=last_user_question,
    )


def _progression_summary(progression: ChordProgression) -> dict[str, Any]:
    roman_numerals = [
        annotation.roman_numeral
        for annotation in progression.chord_annotations
    ]
    return {
        "chords": list(progression.chords),
        "roman_numerals": roman_numerals,
        "score": progression.score,
        "model_confidence": progression.model_confidence,
        "mood_alignment": progression.mood_alignment,
        "harmonic_function": progression.harmonic_function,
        "explanation": progression.explanation,
        "chord_annotations": [
            {
                "position": annotation.position,
                "symbol": annotation.symbol,
                "roman_numeral": annotation.roman_numeral,
                "function_label": annotation.function_label,
                "alignment_percentage": annotation.alignment_percentage,
                "template_phrase": annotation.template_phrase,
            }
            for annotation in progression.chord_annotations
        ],
    }


def _last_refinement_results(user_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": user_params.get("last_refinement"),
        "target": user_params.get("last_refinement_target"),
        "plan": user_params.get("last_refinement_plan"),
        "interpretation": user_params.get("last_refinement_interpretation"),
    }


def _derive_syllable_targets(previous_lyrics: str) -> list[int]:
    return [
        _count_syllables(line)
        for line in previous_lyrics.splitlines()
        if line.strip()
    ]


def _count_syllables(text: str) -> int:
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


def _chat_messages_to_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    return [_model_to_dict(message) for message in messages]


def _infer_last_user_question(chat_history: Sequence[ChatMessage]) -> str | None:
    for message in reversed(chat_history):
        if message.role == "user":
            return message.content
    return None


def _last_items(items: Sequence[Any], limit: int) -> Sequence[Any]:
    if limit == 0:
        return []
    return items[-limit:]


def _model_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value
