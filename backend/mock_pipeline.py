"""Deterministic mock generators used by the early session-centric API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Iterable

from shared.schemas import (
    Action,
    ChatMessage,
    EmotionVector,
    ExplanationReport,
    LyricSuggestion,
    MelodyNote,
    MelodyProfile,
    MelodySuggestion,
    NoteEvent,
    Progression,
    RecognisedChord,
    RefinementPlan,
    SessionState,
)

_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def pitch_to_midi(pitch: str) -> int:
    note = pitch[:-1]
    octave = int(pitch[-1])
    return (octave + 1) * 12 + _PITCH_CLASS[note]


def melody_notes_to_events(notes: list[MelodyNote]) -> list[NoteEvent]:
    return [
        NoteEvent(
            pitch=pitch_to_midi(note.pitch),
            onset=note.start_beat,
            duration=note.duration_beats,
            velocity=note.velocity,
            confidence=0.92,
        )
        for note in notes
    ]


def build_melody_profile(notes: list[NoteEvent]) -> MelodyProfile:
    if not notes:
        return MelodyProfile(
            interval_histogram=[1.0, 0.0, 0.0, 0.0, 0.0],
            rhythmic_density=0.0,
            pitch_range=(0, 0),
            contour="rising",
        )

    ordered = sorted(notes, key=lambda note: note.onset)
    pitches = [note.pitch for note in ordered]
    intervals = [curr - prev for prev, curr in zip(pitches, pitches[1:])]
    buckets = [0, 0, 0, 0, 0]
    for interval in intervals:
        if interval <= -3:
            buckets[0] += 1
        elif interval < 0:
            buckets[1] += 1
        elif interval == 0:
            buckets[2] += 1
        elif interval < 3:
            buckets[3] += 1
        else:
            buckets[4] += 1

    total_intervals = sum(buckets) or 1
    total_beats = max(note.onset + note.duration for note in ordered)
    return MelodyProfile(
        interval_histogram=[bucket / total_intervals for bucket in buckets],
        rhythmic_density=len(ordered) / max(total_beats, 1.0),
        pitch_range=(min(pitches), max(pitches)),
        contour=classify_contour(pitches),
    )


def classify_contour(pitches: list[int]) -> str:
    if len(pitches) < 3:
        return "rising" if pitches[-1] >= pitches[0] else "falling"

    peak = max(range(len(pitches)), key=pitches.__getitem__)
    valley = min(range(len(pitches)), key=pitches.__getitem__)
    if 0 < peak < len(pitches) - 1 and pitches[0] < pitches[peak] and pitches[-1] < pitches[peak]:
        return "arch"
    if 0 < valley < len(pitches) - 1 and pitches[0] > pitches[valley] and pitches[-1] > pitches[valley]:
        return "valley"
    return "rising" if pitches[-1] >= pitches[0] else "falling"


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


def detect_mock_key(mood: str | None) -> str:
    return "A minor" if (mood or "uplift").lower() == "melancholic" else "C major"


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


def build_recognised_chords(state: SessionState) -> list[RecognisedChord]:
    start_points = [0.0, 2.0, 4.0]
    symbols = ["Am", "F", "G"]
    functions = ["tonic substitute", "predominant", "dominant"]
    recognised: list[RecognisedChord] = []
    for index, symbol in enumerate(symbols):
        recognised.append(
            RecognisedChord(
                symbol=symbol,
                start_beat=start_points[index],
                duration_beats=2.0,
                confidence=max(0.62, 0.86 - (index * 0.08)),
                harmonic_function=functions[index],
                explanation=f"Mock BACHI-style decoding favoured {symbol} from the strongest melody tones in this span.",
                decoding_trace=[
                    {
                        "step": "boundary",
                        "confidence": round(0.88 - (index * 0.05), 2),
                        "alternatives": ["hold", "split"],
                    },
                    {
                        "step": "root",
                        "winner": symbol[0],
                        "alternatives": [symbol[0], "C", "E"],
                    },
                    {
                        "step": "quality",
                        "winner": symbol[1:] or "maj",
                        "alternatives": [symbol[1:] or "maj", "7", "sus2"],
                    },
                ],
            )
        )
    return recognised


def build_refinement_plan(instruction: str, target: str | None = None) -> RefinementPlan:
    lowered = instruction.lower()
    target_pipeline = "session"
    adjustments: dict[str, object] = {"requested_target": target or "session"}

    if target == "chords" or any(word in lowered for word in ("jazz", "chord", "harmon")):
        target_pipeline = "harmonizer"
        adjustments["chord_complexity"] = "high" if "jazz" in lowered else "medium"
        adjustments["emotion_valence"] = -0.2 if any(word in lowered for word in ("dark", "sad", "myster")) else 0.1
    elif target == "melody" or any(word in lowered for word in ("melody", "hook", "contour")):
        target_pipeline = "melody_generator"
        adjustments["contour"] = "rising" if "lift" in lowered or "up" in lowered else "varied"
        adjustments["rhythmic_density"] = "lighter" if "space" in lowered else "steady"
    elif target == "suggestions":
        target_pipeline = "both"
        adjustments["variation"] = "higher"
    elif target == "lyrics":
        target_pipeline = "lyric_generator"
        adjustments["imagery"] = "more vivid"

    interpretation = (
        f"Interpreted '{instruction}' as a request to adjust the {target_pipeline.replace('_', ' ')} settings "
        f"while keeping the current session context."
    )
    return RefinementPlan(
        target_pipeline=target_pipeline,
        parameter_adjustments=adjustments,
        interpretation=interpretation,
    )


def build_explanation_report(
    state: SessionState,
    *,
    source_action: str,
    summary: str,
    use_case: str | None = None,
) -> ExplanationReport:
    melody_confidence = [
        {
            "pitch": note.pitch,
            "onset": note.onset,
            "confidence": note.confidence,
        }
        for note in state.melody_notes
    ]
    decoding_traces = [
        {
            "symbol": chord.symbol,
            "start_beat": chord.start_beat,
            "confidence": chord.confidence,
            "trace": chord.decoding_trace,
        }
        for chord in state.recognised_chords
    ]
    constraint_logs = [
        {
            "candidate": index + 1,
            "status": "accepted",
            "reason": suggestion.explanation,
            "score": suggestion.coherence_score,
        }
        for index, suggestion in enumerate(state.melody_suggestions)
    ]
    chord_theory = [
        {
            "progression": progression.chords,
            "harmonic_function": progression.harmonic_function,
            "explanation": progression.explanation,
        }
        for progression in state.chord_progressions
    ]
    emotion_mapping = (
        state.emotion_vector.model_dump()
        if state.emotion_vector is not None
        else {}
    )
    cache_status = {"use_case": use_case, "cache_hit": False} if use_case else {}
    return ExplanationReport(
        source_action=source_action,
        summary=summary,
        melody_confidence=melody_confidence,
        decoding_traces=decoding_traces,
        constraint_logs=constraint_logs,
        chord_theory=chord_theory,
        emotion_mapping=emotion_mapping,
        cache_status=cache_status,
    )


def build_chat_reply(state: SessionState, message: str) -> ChatMessage:
    summary = (
        state.explanation_report.summary
        if state.explanation_report is not None
        else "There is no explanation report yet, so this response stays at the session-summary level."
    )
    chord_detail = ""
    if state.recognised_chords:
        chord = state.recognised_chords[0]
        chord_detail = f" The current trace starts with {chord.symbol} at beat {chord.start_beat:.1f}."
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
