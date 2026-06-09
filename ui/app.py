"""Streamlit UI shell for the HumMuse co-creative workflow."""

from __future__ import annotations

import copy
import os
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st

try:
    from ui.audio_utils import (
        decode_midi_bytes,
        render_audio_from_session,
        synthesize_progression_audio,
        synthesize_wave_from_notes,
    )
    from ui.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from audio_utils import (
        decode_midi_bytes,
        render_audio_from_session,
        synthesize_progression_audio,
        synthesize_wave_from_notes,
    )
    from i18n import DEFAULT_LANGUAGE, LANGUAGES, t


DEFAULT_API_URL = os.environ.get("HUMMUSE_API_URL", "http://127.0.0.1:8000")
SOUNDFONT_PATH = os.environ.get("HUMMUSE_SOUNDFONT", "")
SUPPORTED_AUDIO_TYPES = ["wav", "webm", "mp3"]
_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Song Brief (creative-context graph) ------------------------------------
# The song-level emotional brief: the author's mood choice flows into
# downstream subsystems (DQN chord head, transformer melody continuation,
# GPT lyric prompts) via state.emotion_vector, persisted through
# POST /session/mood. Scope is deliberately narrow: only fields the backend
# actually consumes today get a widget. Genre / Theme / Vocabulary are not
# wired end-to-end, so they are intentionally absent — the dependency graph
# must remain honest, not aspirational.
#
# MOOD_OPTIONS is the offline fallback for the canonical mood taxonomy; the
# live list is fetched from GET /moods (see fetch_mood_options). Keep this
# mirror in sync with shared.schemas.EMOTION_PRESET_LABELS — a test pins it.
MOOD_OPTIONS = [
    "joyful",
    "triumphant",
    "uplift",
    "hopeful",
    "romantic",
    "calm",
    "neutral",
    "reflective",
    "melancholic",
    "dark",
    "anxious",
    "tense",
]
MOOD_DEFAULT = "uplift"
# Legacy/aliased mood labels from older sessions/snapshots → canonical preset.
MOOD_ALIASES = {"melancholy": "melancholic"}
# Lyric generation styles. Values must match VALID_LYRIC_MODES in
# ml/gpt/prompts/lyrics.py — the backend silently maps anything else
# to "Poetic" (the default), which is exactly the bug a missing mode
# selector produces. Keep this list synchronized if the backend grows
# new modes.
LYRIC_MODE_OPTIONS = ["Simpler", "Poetic", "Catchy"]
LYRIC_MODE_DEFAULT = "Poetic"
DEFAULT_SONG_BRIEF: dict[str, Any] = {
    "mood": MOOD_DEFAULT,
}


def api_get(api_base_url: str, path: str) -> Any:
    response = requests.get(f"{api_base_url.rstrip('/')}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(api_base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.post(f"{api_base_url.rstrip('/')}{path}", json=payload or {}, timeout=15)
    response.raise_for_status()
    return response.json()


def api_patch(api_base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.patch(f"{api_base_url.rstrip('/')}{path}", json=payload or {}, timeout=15)
    response.raise_for_status()
    return response.json()


def api_post_multipart(
    api_base_url: str,
    path: str,
    *,
    files: dict[str, Any],
    data: dict[str, Any],
) -> Any:
    response = requests.post(
        f"{api_base_url.rstrip('/')}{path}",
        files=files,
        data=data,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_get_json_by_url(url: str) -> Any:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def load_sessions(api_base_url: str) -> list[dict[str, Any]]:
    return api_get(api_base_url, "/sessions")


def create_session(api_base_url: str) -> dict[str, Any]:
    return api_post(api_base_url, "/session/create")


def fetch_session_state(api_base_url: str, session_id: str) -> dict[str, Any]:
    return api_get(api_base_url, f"/session/{session_id}/state")["state"]


def build_audio_file_tuple(uploaded_audio: Any) -> tuple[str, bytes, str]:
    filename = getattr(uploaded_audio, "name", None) or "hum.wav"
    mime_type = getattr(uploaded_audio, "type", None) or "audio/wav"
    if hasattr(uploaded_audio, "getvalue"):
        content = uploaded_audio.getvalue()
    else:
        content = uploaded_audio.read()
    return filename, content, mime_type


def extract_melody_from_audio(
    api_base_url: str,
    *,
    session_id: str,
    uploaded_audio: Any,
) -> dict[str, Any]:
    """POST audio to the melody-from-hum endpoint.

    ``prompt``, ``tempo_bpm`` and ``mood`` are deliberately not sent:
    ``prompt`` was a no-op annotation, ``tempo_bpm`` should be detected
    from audio, and ``mood`` only nudged the *mock* melody's final pitch —
    it is not the canonical emotion lever (that is the Song Brief →
    state.emotion_vector path). The endpoint still accepts all three as
    optional Form fields, so omitting them is non-breaking.
    """
    audio_file = build_audio_file_tuple(uploaded_audio)
    return api_post_multipart(
        api_base_url,
        "/melody/from-hum",
        files={"audio": audio_file},
        data={
            "session_id": session_id,
        },
    )


def note_rows(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pitch": note["pitch"],
            "pitch_label": format_midi_pitch(int(note["pitch"])),
            "onset": note["onset"],
            "duration": note["duration"],
            "velocity": note["velocity"],
            "confidence": note["confidence"],
        }
        for note in notes
    ]


def format_midi_pitch(midi_pitch: int) -> str:
    octave = (int(midi_pitch) // 12) - 1
    note_name = _PITCH_CLASS_NAMES[int(midi_pitch) % 12]
    return f"{note_name}{octave} ({int(midi_pitch)})"


def _friendly_action_label(source_action: str) -> str:
    """Translate a backend `source_action` slug to a human label.

    Tries `t("explanations.action.<slug>")` first; if no translation
    exists (i.e. t() returns the key unchanged — its loud-failure
    behaviour), falls back to a cleaned-up display of the slug so
    unknown actions still render readably instead of as `key.path`.
    """
    key = f"explanations.action.{source_action}"
    label = t(key)
    if label == key:
        return source_action.replace("_", " ").capitalize()
    return label


def _short_model_label(model: str) -> str:
    """Trim `gpt://<id>/<family>/<version>@<rev>` to `<family>/<version>`.

    The full identifier is fine for audit logs but useless visually.
    Defensive against unknown shapes — returns the original string with
    revision lopped off if the pattern doesn't match.
    """
    base = model.split("@", 1)[0]
    parts = [p for p in base.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return base


def normalize_mood(label: str | None) -> str | None:
    """Map a mood label to a canonical preset, or ``None`` if unrecognised.

    Mirrors ``shared.schemas.normalize_mood_label`` client-side (the UI cannot
    import backend/shared). Applies known aliases (``melancholy`` →
    ``melancholic``) and returns ``None`` for labels outside the fetched/known
    taxonomy so the caller can fall back to a safe default.
    """
    if not label:
        return None
    canonical = MOOD_ALIASES.get(label, label)
    return canonical if canonical in mood_options() else None


def fetch_mood_options(api_base_url: str) -> list[str]:
    """Return the canonical mood labels from the backend, cached per session.

    Falls back to the mirrored ``MOOD_OPTIONS`` constant if the backend is
    unreachable so the Song Brief still renders offline.
    """
    cached = st.session_state.get("mood_options")
    if cached:
        return cached
    try:
        labels = api_get(api_base_url, "/moods").get("labels") or []
    except requests.RequestException:
        labels = []
    options = labels or list(MOOD_OPTIONS)
    st.session_state["mood_options"] = options
    return options


def mood_options() -> list[str]:
    """The currently-known mood labels (fetched cache or offline fallback)."""
    return st.session_state.get("mood_options") or list(MOOD_OPTIONS)


def set_session_mood(api_base_url: str, *, session_id: str, mood: str) -> dict[str, Any]:
    """Persist an author-chosen Song Brief mood as the session emotion."""
    return api_post(
        api_base_url,
        "/session/mood",
        payload={"session_id": session_id, "mood": mood},
    )


def artifact_by_kind(artifacts: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for artifact in artifacts:
        if artifact.get("kind") == kind:
            return artifact
    return None


def extract_pipeline_timings(response: dict[str, Any]) -> dict[str, Any] | None:
    artifact = artifact_by_kind(response.get("artifacts", []), "timings")
    if artifact is None or not artifact.get("url"):
        return None
    return api_get_json_by_url(str(artifact["url"]))


def generate_chords_from_lyrics(
    api_base_url: str,
    *,
    session_id: str,
    lyrics_text: str,
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/chords/from-lyrics",
        payload={"session_id": session_id, "text": lyrics_text},
    )


def generate_chords_from_melody(
    api_base_url: str,
    *,
    session_id: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the DQN harmonizer purely from the session melody.

    Backend requires ``state.melody_notes`` to be populated (UI should
    gate the button on it). ``emotion_vector`` is omitted so the
    endpoint falls back to ``state.emotion_vector`` if present, else
    the neutral default — exactly the contract documented on
    ``/chords/from-melody``.
    """
    return api_post(
        api_base_url,
        "/chords/from-melody",
        payload={"session_id": session_id, "top_k": top_k},
    )


def submit_manual_chords(
    api_base_url: str,
    *,
    session_id: str,
    chords: list[str],
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/chords/manual",
        payload={"session_id": session_id, "chords": chords},
    )


def parse_manual_chord_input(raw: str) -> list[str]:
    """Split free-form chord input on commas, arrows, pipes, and whitespace.

    Accepts inputs like "C, F, G, C", "C -> F -> G -> C", "C | F | G | C",
    or just "C F G C". Returns the cleaned, non-empty token list. Validation
    of individual symbols is the API's job (it reuses the constraint
    pipeline's parser so the rules cannot drift between layers).
    """
    if not raw:
        return []
    import re

    return [token for token in re.split(r"[\s,|]+|->|—|–", raw) if token]


def request_melody_suggestions(
    api_base_url: str,
    *,
    session_id: str,
    num_suggestions: int = 3,
    primer_section: str | None = None,
    target_section: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "num_suggestions": num_suggestions,
    }
    # Only include section labels when actually set; the backend treats
    # missing fields as "fall back to primer-relative constraints".
    if primer_section is not None:
        payload["primer_section"] = primer_section
    if target_section is not None:
        payload["target_section"] = target_section
    return api_post(
        api_base_url,
        "/melody/continue",
        payload=payload,
    )


def accept_melody_suggestion(
    api_base_url: str,
    *,
    session_id: str,
    suggestion_index: int,
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/melody/accept",
        payload={"session_id": session_id, "suggestion_index": suggestion_index},
    )


def request_lyric_suggestions(
    api_base_url: str,
    *,
    session_id: str,
    mode: str = LYRIC_MODE_DEFAULT,
    num_suggestions: int = 3,
    lyrics_text: str | None = None,
) -> dict[str, Any]:
    """Trigger lyric-suggestion generation.

    Default ``mode`` matches the backend's silent fallback ("Poetic")
    so callers that forget to pass one don't surprise the user with a
    style they didn't ask for. ``lyrics_text`` is sent only when
    non-empty so older backends that ignore unknown fields stay happy
    and we don't accidentally wipe session state with an empty buffer
    from a stale text area.
    """
    payload: dict[str, Any] = {
        "session_id": session_id,
        "mode": mode,
        "num_suggestions": num_suggestions,
    }
    if lyrics_text and lyrics_text.strip():
        payload["lyrics_text"] = lyrics_text
    return api_post(api_base_url, "/suggest/lyrics", payload=payload)


def refine_session(
    api_base_url: str,
    *,
    session_id: str,
    instruction: str,
    target: str | None = None,
) -> dict[str, Any]:
    return api_patch(
        api_base_url,
        f"/session/{session_id}/refine",
        payload={"instruction": instruction, "target": target},
    )


def send_chat_message(
    api_base_url: str,
    *,
    session_id: str,
    message: str,
    language: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if language:
        payload["language"] = language
    return api_post(
        api_base_url,
        f"/session/{session_id}/chat",
        payload=payload,
    )


def _compress_chord_sequence(chords: list[str], *, separator: str = " -> ") -> str:
    """Run-length encode adjacent duplicate chord symbols.

    ``["C", "C", "C", "F", "G"]`` → ``"C ×3 -> F -> G"``. Keeps the
    progression title readable when the backend returns long runs of
    the same chord (e.g. an 18-position DQN output of all "Caug")
    instead of scaring the user with a wall of identical symbols.
    Single occurrences render without the multiplier suffix.
    """
    if not chords:
        return ""
    parts: list[str] = []
    prev = chords[0]
    count = 1
    for chord in chords[1:]:
        if chord == prev:
            count += 1
            continue
        parts.append(f"{prev} ×{count}" if count > 1 else prev)
        prev = chord
        count = 1
    parts.append(f"{prev} ×{count}" if count > 1 else prev)
    return separator.join(parts)


def progression_title(progression: dict[str, Any]) -> str:
    return _compress_chord_sequence(progression.get("chords") or [])


def progression_caption(progression: dict[str, Any]) -> str:
    """Short, user-facing summary of a progression — scores + function.

    The verbose backend ``explanation`` field (rollout details, reward
    breakdowns, DQN epoch tags, etc.) is intentionally excluded here.
    It is technical audit data, not a card caption — it lives behind
    the "Show technical details" expander in the chord card.
    """
    model_confidence = progression.get("model_confidence", progression.get("score"))
    mood_alignment = progression.get("mood_alignment")
    harmonic_function = progression.get("harmonic_function")
    labels: list[str] = []
    if model_confidence is not None:
        labels.append(f"Model confidence: {float(model_confidence):.2f}")
    if mood_alignment is not None:
        labels.append(f"Mood alignment: {float(mood_alignment):.2f}")
    if harmonic_function:
        labels.append(str(harmonic_function))
    return " | ".join(labels)


def top_pitch_class_probs(distribution: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    policy = distribution.get("policy", distribution)
    pitch_class = policy.get("pitch_class", {})
    ranked = sorted(pitch_class.items(), key=lambda item: item[1], reverse=True)
    return [
        {"pitch_class": name, "probability": round(float(probability), 4)}
        for name, probability in ranked[:limit]
    ]


def render_progression_details(progression: dict[str, Any]) -> None:
    """Render the diagnostic detail for one progression.

    Flattened — no inner expanders — because callers wrap this in an
    outer "Show technical details" expander. Streamlit allows nested
    expanders but they read poorly; one outer gate over plain headed
    sections is cleaner.
    """
    annotations = progression.get("chord_annotations", [])
    distributions = progression.get("native_distributions", [])
    if annotations:
        st.markdown(f"**{t('chords.annotations_header')}**")
        for annotation in annotations:
            st.markdown(
                f"**{annotation['position'] + 1}. {annotation['symbol']}** "
                f"({annotation['roman_numeral']}, {annotation['function_label']})"
            )
            st.caption(
                f"Alignment: {annotation['alignment_percentage']:.2f} | "
                f"{annotation['template_phrase']}"
            )
            strong_notes = annotation.get("strong_beat_notes", [])
            if strong_notes:
                st.json(strong_notes)
    if distributions:
        st.markdown(f"**{t('chords.distributions_header')}**")
        for distribution in distributions:
            policy = distribution.get("policy", distribution)
            st.markdown(f"**Position {distribution['position'] + 1}**")
            st.write(
                {
                    "rest": policy.get("rest", {}),
                    "octave": policy.get("octave", {}),
                    "inversion": policy.get("inversion", {}),
                    "q_margin": distribution.get("q_margin", {}),
                    "emotion_bias": distribution.get("emotion_bias"),
                    "top_pitch_classes": top_pitch_class_probs(distribution),
                }
            )


def numbered_label(index: int, text: str) -> str:
    return f"{index}. {text}"


def render_grid(
    items: list[Any],
    cols_per_row: int,
    render_card: Any,
) -> None:
    """Render `items` as a wrapping grid of equal-width cards.

    Each row holds up to `cols_per_row` columns; empty trailing cells
    in the last row stay blank so card widths are stable across rows.
    `render_card(index, item)` runs inside each column's `with` block —
    pass a closure that captures the surrounding state if you need it.

    Used to honor the "multiple variants as cards" UX principle: melody
    continuations and lyric suggestions get N=3 (v0.dev convention),
    chord progressions get N=4 (Midjourney convention). The old artifact
    isn't destroyed when a card is applied — the version slider keeps
    the prior state recoverable.
    """
    if not items:
        return
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for offset, item in enumerate(row_items):
            with cols[offset]:
                render_card(row_start + offset, item)


def confidence_to_color(confidence: float) -> str:
    if confidence >= 0.85:
        return "#54AC3C"
    if confidence >= 0.7:
        return "#ead02d"
    return "#e74f4f"


def render_confidence_note_table(notes: list[dict[str, Any]]) -> None:
    rows = [
        "<tr>"
        f"<td>{note['pitch_label']}</td>"
        f"<td>{note['onset']:.2f}</td>"
        f"<td>{note['duration']:.2f}</td>"
        f"<td>{note['velocity']}</td>"
        f"<td style='background:{confidence_to_color(float(note['confidence']))};'>{float(note['confidence']):.2f}</td>"
        "</tr>"
        for note in notes
    ]
    header_html = (
        '<table style="width:100%; border-collapse:collapse;">'
        "<thead><tr>"
        f'<th style="text-align:left;">{t("notes.pitch")}</th>'
        f'<th style="text-align:left;">{t("notes.onset")}</th>'
        f'<th style="text-align:left;">{t("notes.duration")}</th>'
        f'<th style="text-align:left;">{t("notes.velocity")}</th>'
        f'<th style="text-align:left;">{t("notes.confidence")}</th>'
        "</tr></thead><tbody>"
    )
    st.markdown(
        header_html + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )


def format_session_option(session: dict[str, Any]) -> str:
    session_id = session["session_id"]
    updated_at = session["updated_at"].replace("T", " ").replace("Z", " UTC")
    return f"{session_id} | updated {updated_at}"


def ensure_session_defaults() -> None:
    st.session_state.setdefault("sessions_cache", [])
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("active_session_state", None)
    st.session_state.setdefault("selected_session_label", None)
    st.session_state.setdefault("lyrics_input", "")
    st.session_state.setdefault("audio_source", None)
    st.session_state.setdefault("melody_refine_text", "")
    st.session_state.setdefault("chords_refine_text", "")
    st.session_state.setdefault("lyrics_refine_text", "")
    st.session_state.setdefault("explanations_chat_text", "")
    st.session_state.setdefault("melody_playback_audio", None)
    st.session_state.setdefault("melody_playback_source", None)
    st.session_state.setdefault("latest_melody_result", None)
    st.session_state.setdefault("latest_melody_timings", None)
    st.session_state.setdefault("snapshots", [])
    st.session_state.setdefault("song_brief", copy.deepcopy(DEFAULT_SONG_BRIEF))
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)


def clear_playback_cache() -> None:
    st.session_state.melody_playback_audio = None
    st.session_state.melody_playback_source = None


def reset_history(initial_key: str, **kwargs: Any) -> None:
    """Reset the snapshot timeline for a freshly created/loaded session.

    Called when the active session changes — we don't carry snapshots
    across sessions. The first entry is the as-loaded state so the
    timeline always has a base to roll back to.
    """
    st.session_state.snapshots = []
    # Drop the version-slider's persisted selection. Its widget key holds a
    # label string from the *previous* session's snapshots; if the new session
    # later grows enough snapshots to re-render the slider, Streamlit would
    # validate that stale label against the new options and raise
    # StreamlitAPIException. Clearing it lets the slider fall back to its
    # `value=options[-1]` default on next render.
    st.session_state.pop("history_slider", None)
    push_snapshot(initial_key, **kwargs)


def reset_song_brief() -> None:
    """Reset the Song Brief to defaults — called on session create/load."""
    st.session_state.song_brief = copy.deepcopy(DEFAULT_SONG_BRIEF)


def reset_lyrics_input(state: dict[str, Any] | None) -> None:
    """Seed the Lyrics tab text-area buffer from a (re)loaded session.

    Streamlit ignores a widget's ``value=`` once its ``key`` already
    exists in ``session_state`` (which it always does — we seed it in
    ``ensure_session_defaults``). So pre-population has to happen by
    writing the key directly, *before* the widget is instantiated —
    same approach as ``reset_song_brief``/``reset_history`` on a
    session change. Called on create/load/restore; in-session edits stay
    sticky because nothing rewrites the key between those events.
    """
    st.session_state["lyrics_input"] = (state or {}).get("lyrics_text") or ""


def push_snapshot(label_key: str, **kwargs: Any) -> None:
    """Append a deep-copied snapshot of the active session state.

    Pushed *after* a successful mutation so each snapshot represents the
    artifact as it stood once an action completed. Slider semantics: the
    rightmost snapshot is always the current live state.

    The Song Brief is captured alongside session state because creative
    intent is part of "the version" of the song — restoring an older
    artifact should also restore the creative context that produced it.

    Labels are stored as an i18n key + kwargs pair rather than a
    pre-formatted string so the version slider reflects the active
    language even after the user flips the language toggle.
    """
    state = st.session_state.active_session_state
    if not state:
        return
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_key": label_key,
        "label_kwargs": kwargs,
        "state": copy.deepcopy(state),
        "song_brief": copy.deepcopy(
            st.session_state.get("song_brief", DEFAULT_SONG_BRIEF)
        ),
    }
    st.session_state.snapshots.append(snapshot)


def snapshot_label_text(snapshot: dict[str, Any]) -> str:
    """Resolve the display label for a snapshot in the active language.

    Backward-compatible: in-memory snapshots from an older code path
    that stored a pre-formatted ``label`` string still render.
    """
    key = snapshot.get("label_key")
    if key:
        return t(key, **snapshot.get("label_kwargs", {}))
    return snapshot.get("label", "")


def format_snapshot_option(index: int, snapshot: dict[str, Any]) -> str:
    return t(
        "snapshot.version_label",
        n=index + 1,
        label=snapshot_label_text(snapshot),
    )


def refresh_sessions(api_base_url: str) -> None:
    st.session_state.sessions_cache = load_sessions(api_base_url)


def sync_active_session(api_base_url: str) -> None:
    active_session_id = st.session_state.active_session_id
    if active_session_id:
        st.session_state.active_session_state = fetch_session_state(api_base_url, active_session_id)
        clear_playback_cache()


def render_language_toggle() -> None:
    """Render the EN/RU language switch at the top of the sidebar.

    Stored as ``st.session_state.language``; resolution happens through
    ``ui.i18n.t``. The widget key is the canonical language-state slot
    itself so flipping the radio updates the language on the next
    rerun without an extra assignment.
    """
    st.sidebar.radio(
        t("language.label"),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        horizontal=True,
        key="language",
    )
    st.sidebar.divider()


def render_sidebar() -> None:
    render_language_toggle()
    st.sidebar.title(t("app.title"))
    st.sidebar.caption(t("sidebar.subtitle"))

    api_base_url = DEFAULT_API_URL

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button(t("sidebar.new_session"), use_container_width=True):
        try:
            created = create_session(api_base_url)
            state = created["state"]
            st.session_state.active_session_id = state["session_id"]
            st.session_state.active_session_state = state
            clear_playback_cache()
            reset_song_brief()
            reset_lyrics_input(state)
            reset_history("snapshot.initial")
            refresh_sessions(api_base_url)
            st.session_state.selected_session_label = next(
                (
                    format_session_option(session)
                    for session in st.session_state.sessions_cache
                    if session["session_id"] == state["session_id"]
                ),
                None,
            )
            st.sidebar.success(t("sidebar.created_session", sid=state["session_id"]))
        except requests.RequestException as exc:
            st.sidebar.error(t("sidebar.could_not_create", err=exc))

    if col_b.button(t("sidebar.refresh"), use_container_width=True):
        try:
            refresh_sessions(api_base_url)
            st.sidebar.success(t("sidebar.refreshed"))
        except requests.RequestException as exc:
            st.sidebar.error(t("sidebar.could_not_load_list", err=exc))

    if not st.session_state.sessions_cache:
        try:
            refresh_sessions(api_base_url)
        except requests.RequestException:
            pass

    options = {format_session_option(session): session["session_id"] for session in st.session_state.sessions_cache}
    none_label = t("sidebar.option_none")
    selected_label = st.sidebar.selectbox(
        t("sidebar.load_existing"),
        options=[none_label, *options.keys()],
        index=0 if st.session_state.selected_session_label not in options else list(options.keys()).index(st.session_state.selected_session_label) + 1,
    )

    if st.sidebar.button(t("sidebar.load_session"), use_container_width=True):
        if selected_label == none_label:
            st.sidebar.info(t("sidebar.choose_session_first"))
        else:
            session_id = options[selected_label]
            try:
                loaded_state = fetch_session_state(api_base_url, session_id)
                st.session_state.active_session_state = loaded_state
                st.session_state.active_session_id = session_id
                st.session_state.selected_session_label = selected_label
                clear_playback_cache()
                reset_song_brief()
                reset_lyrics_input(loaded_state)
                reset_history("snapshot.loaded")
                st.sidebar.success(t("sidebar.loaded_session", sid=session_id))
            except requests.RequestException as exc:
                st.sidebar.error(t("sidebar.could_not_load", err=exc))

    st.sidebar.divider()
    active_session_id = st.session_state.active_session_id
    if active_session_id:
        st.sidebar.markdown(f"**{t('sidebar.active_session')}**")
        st.sidebar.code(active_session_id)
        render_history_panel()
    else:
        st.sidebar.info(t("sidebar.create_or_load"))


def render_history_panel() -> None:
    """Render the version slider + restore control in the sidebar.

    The rightmost snapshot is the current live state; selecting an
    earlier version and clicking Restore copies its state back into
    `active_session_state` and records the restore as its own snapshot
    (so the timeline is append-only and the user can always wind
    forward again).
    """
    snapshots = st.session_state.get("snapshots", [])
    st.sidebar.markdown(f"**{t('history.title')}**")
    if not snapshots:
        st.sidebar.caption(t("history.no_snapshots"))
        return

    if len(snapshots) == 1:
        st.sidebar.caption(
            t("history.single_snapshot", label=snapshot_label_text(snapshots[0]))
        )
        return

    options = [format_snapshot_option(i, s) for i, s in enumerate(snapshots)]
    selected_label = st.sidebar.select_slider(
        t("history.browse"),
        options=options,
        value=options[-1],
        key="history_slider",
    )
    selected_index = options.index(selected_label)
    selected = snapshots[selected_index]
    is_latest = selected_index == len(snapshots) - 1
    timestamp = selected["timestamp"].replace("T", " ").replace("+00:00", " UTC")
    st.sidebar.caption(t("history.saved_at", time=timestamp))
    if is_latest:
        st.sidebar.caption(t("history.this_is_live"))
    else:
        st.sidebar.caption(
            t("history.viewing", n=selected_index + 1, total=len(snapshots))
        )

    if st.sidebar.button(
        t("history.restore"),
        use_container_width=True,
        disabled=is_latest,
        help=t("history.restore_help"),
    ):
        restored_state = copy.deepcopy(selected["state"])
        st.session_state.active_session_state = restored_state
        # Read the new "song_brief" snapshot key, falling back to the pre-rename
        # "story_bible" key so snapshots captured before the rename still restore.
        st.session_state.song_brief = copy.deepcopy(
            selected.get("song_brief", selected.get("story_bible", DEFAULT_SONG_BRIEF))
        )
        reset_lyrics_input(restored_state)
        clear_playback_cache()
        push_snapshot(
            "snapshot.restored",
            n=selected_index + 1,
            label=snapshot_label_text(selected),
        )
        st.sidebar.success(t("history.restored", n=selected_index + 1))
        st.rerun()


def render_header() -> None:
    # set_page_config runs once per session; the browser tab title
    # therefore won't change mid-session if the user flips languages —
    # we keep the brand name "HumMuse" which is universal anyway.
    st.set_page_config(page_title="HumMuse", page_icon="🎼", layout="wide")
    st.title(t("app.title"))
    st.caption(t("app.caption"))


def build_song_brief_dot(brief: dict[str, Any]) -> str:
    """Build a Graphviz DOT diagram of the Song Brief dependency graph.

    Nodes are colored by role: yellow for user-controlled creative
    intent, blue for fields auto-detected from the user's audio, green
    for generative subsystems that consume them. Edge labels name what
    flows along each edge — every edge corresponds to a real code path
    in the repo. The Mood node is authoritative: picking a mood calls
    POST /session/mood, which sets state.emotion_vector, and all three
    subsystems already read that vector:

        Mood -> DQN          → POST /session/mood → state.emotion_vector → ml/harmony/emotion_modulation.py
        Mood -> Transformer  → state.emotion_vector (arousal) → ml/melody_sketchpad/continuation/emotion_temperature.py
        Mood -> GPT          → state.emotion_vector → emotion_vector field in ml/gpt/prompts/lyrics.py
        Lyrics -> Mood       → backend/mood_responder.py (suggest, not override)
        Lyrics -> GPT        → previous_lyrics field in lyric prompt
        Key   -> Transformer → continuation constraints
        Tempo -> Transformer → continuation constraints

    Nothing aspirational lives here — the graph is the explainability
    contract.
    """
    mood = brief.get("mood", "—")
    return f"""
    digraph SongBrief {{
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", fontsize=9, color="#475569"];

        // User-controlled creative intent (yellow)
        Mood   [label="Mood\\n({mood})", fillcolor="#fde68a"];
        Lyrics [label="Lyrics",          fillcolor="#fde68a"];

        // Auto-detected from audio (blue)
        Key   [label="Detected key",   fillcolor="#bae6fd"];
        Tempo [label="Detected tempo", fillcolor="#bae6fd"];

        // Generative subsystems (green)
        DQN         [label="DQN chord\\ncandidates",            fillcolor="#d9f99d"];
        Transformer [label="Transformer melody\\ncontinuation", fillcolor="#d9f99d"];
        GPT         [label="GPT lyric\\nsuggestions",           fillcolor="#d9f99d"];

        Mood   -> DQN         [label="emotion vector"];
        Mood   -> Transformer [label="temperature (arousal)"];
        Mood   -> GPT         [label="emotion vector"];
        Lyrics -> Mood        [style=dashed, label="auto-detect (suggestion)"];
        Lyrics -> GPT         [label="previous lyrics"];
        Key    -> Transformer [label="key constraint"];
        Tempo  -> Transformer [label="tempo constraint"];
    }}
    """


def render_song_brief() -> None:
    """Render the Song Brief panel: the editable mood (creative intent),
    read-only detected fields, and a dependency graph showing which
    fields drive which subsystems.

    The mood is the single source of truth for the session's emotion:
    selecting one calls POST /session/mood, which persists
    state.emotion_vector server-side and marks it "authored". Changes are
    captured in the next version snapshot and restored alongside the
    artifact when the user rolls back via the slider.
    """
    state = st.session_state.active_session_state
    if not state:
        return
    brief = st.session_state.song_brief
    options = fetch_mood_options(DEFAULT_API_URL)

    with st.expander(t("brief.expander_title"), expanded=True):
        st.caption(t("brief.caption"))

        # Editable creative intent. Values sent to the backend stay English
        # (the backend keys on them); only the display label changes per
        # language. Unknown/legacy labels fall back to "neutral" rather than
        # index 0 so a stale value degrades to "no signal".
        current = normalize_mood(brief.get("mood")) or (
            "neutral" if "neutral" in options else options[0]
        )
        mood_index = options.index(current)
        selected_mood = st.selectbox(
            t("brief.mood_label"),
            options=options,
            index=mood_index,
            key="song_brief_mood",
            format_func=lambda m: t(f"mood.{m}"),
            help=t("brief.mood_help"),
        )
        if selected_mood != current or brief.get("mood") != selected_mood:
            brief["mood"] = selected_mood
            # Persist as the authoritative session emotion. Only fire the call
            # when the value actually changed from what the backend holds.
            if selected_mood != state.get("mood_label"):
                try:
                    set_session_mood(
                        DEFAULT_API_URL,
                        session_id=state["session_id"],
                        mood=selected_mood,
                    )
                    st.session_state.active_session_state = fetch_session_state(
                        DEFAULT_API_URL, state["session_id"]
                    )
                    push_snapshot("snapshot.mood_set", mood=t(f"mood.{selected_mood}"))
                except requests.RequestException as exc:
                    st.error(t("brief.could_not_set_mood", err=exc))

        # Provenance: authored (chosen here) vs detected (inferred from lyrics).
        emotion_source = state.get("emotion_source")
        if emotion_source == "authored":
            st.caption(t("brief.mood_authored"))
        elif emotion_source == "detected":
            st.caption(t("brief.mood_detected"))

        # Read-only auto-detected fields
        st.markdown(t("brief.detected_header"))
        detected_key = state.get("detected_key") or "—"
        detected_tempo = state.get("detected_tempo")
        tempo_str = (
            f"{float(detected_tempo):.0f} BPM" if detected_tempo is not None else "—"
        )
        meta_a, meta_b = st.columns(2)
        meta_a.metric(t("brief.detected_key"), detected_key)
        meta_b.metric(t("brief.detected_tempo"), tempo_str)

        # Dependency graph — the thesis-defensibility piece. Graph
        # contents stay English-labeled by design: the labels correspond
        # to code identifiers (DQN, BiMMuDa, etc.) and code paths in the
        # repo, so a translated graph would mislead an examiner reading
        # the source.
        # Streamlit forbids nested expanders, so use a bordered container.
        st.markdown(f"**{t('brief.graph_expander')}**")
        with st.container(border=True):
            st.caption(t("brief.graph_caption"))
            st.graphviz_chart(build_song_brief_dot(brief), use_container_width=True)


def render_session_overview() -> None:
    state = st.session_state.active_session_state
    if not state:
        st.info(t("overview.use_sidebar"))
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(t("overview.session"), state["session_id"][:8])
    col_b.metric(t("overview.melody_notes"), len(state.get("melody_notes", [])))
    col_c.metric(t("overview.lyric_suggestions"), len(state.get("lyric_suggestions", [])))
    col_d.metric(t("overview.chat_turns"), len(state.get("chat_history", [])))

    st.caption(t("overview.current_session_id", sid=state["session_id"]))
    last_refinement = state.get("user_params", {}).get("last_refinement")
    if last_refinement:
        target_code = state.get("user_params", {}).get("last_refinement_target", "session")
        target_label = t(f"target.{target_code}")
        st.caption(t("overview.latest_refinement", target=target_label, text=last_refinement))
    interpretation = state.get("user_params", {}).get("last_refinement_interpretation")
    if interpretation:
        st.caption(t("overview.interpretation", text=interpretation))


def render_melody_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader(t("melody.subheader"))
    if not state:
        st.write(t("melody.context_placeholder"))
        return

    st.write(t("melody.upload_label"))
    # Track which input was touched most recently. `on_change` fires for
    # both "value set" and "value cleared", so the simple latest-wins
    # rule below also handles users clearing one input to fall back to
    # the other. Default precedence (no interaction yet) is recording
    # over upload, matching the historical behaviour for fresh sessions.
    def _mark_recording_source() -> None:
        st.session_state.audio_source = "recording"

    def _mark_upload_source() -> None:
        st.session_state.audio_source = "upload"

    recorded_audio = st.audio_input(
        t("melody.record"),
        key="hum_recording",
        on_change=_mark_recording_source,
    )
    uploaded_audio = st.file_uploader(
        t("melody.upload"),
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        help=t("melody.upload_help"),
        key="hum_upload",
        on_change=_mark_upload_source,
    )

    # Resolve the active source: respect the user's most recent action,
    # but if that source got cleared, fall back to the other one rather
    # than blocking the Extract button.
    active_source = st.session_state.get("audio_source")
    if active_source == "upload":
        chosen_audio = uploaded_audio or recorded_audio
    elif active_source == "recording":
        chosen_audio = recorded_audio or uploaded_audio
    else:
        chosen_audio = recorded_audio or uploaded_audio

    if chosen_audio is not None:
        _, audio_bytes, mime_type = build_audio_file_tuple(chosen_audio)
        st.audio(audio_bytes, format=mime_type)
        # Surface which source we'll actually extract from — otherwise
        # users with both inputs filled can't tell which one wins.
        if chosen_audio is uploaded_audio:
            st.caption(
                t(
                    "melody.source_upload",
                    name=getattr(uploaded_audio, "name", "audio"),
                )
            )
        else:
            st.caption(t("melody.source_recording"))

    if st.button(t("melody.extract_button"), use_container_width=True, disabled=chosen_audio is None):
        try:
            response = extract_melody_from_audio(
                DEFAULT_API_URL,
                session_id=state["session_id"],
                uploaded_audio=chosen_audio,
            )
            st.session_state.latest_melody_result = response
            try:
                st.session_state.latest_melody_timings = extract_pipeline_timings(response)
            except requests.RequestException:
                st.session_state.latest_melody_timings = None
            st.session_state.active_session_state = fetch_session_state(
                DEFAULT_API_URL,
                response["session_id"],
            )
            clear_playback_cache()
            state = st.session_state.active_session_state
            push_snapshot("snapshot.extracted_melody", n=len(response["melody_notes"]))
            st.success(t("melody.extracted_n", n=len(response["melody_notes"])))
        except requests.RequestException as exc:
            st.error(t("melody.could_not_extract", err=exc))

    notes = state.get("melody_notes", [])
    if notes:
        st.write(t("melody.current_notes"))
        st.caption(t("melody.confidence_legend"))
        render_confidence_note_table(note_rows(notes))
    else:
        st.write(t("melody.no_notes_yet"))

    # melody_profile (pitch class histogram, interval distribution, contour
    # stats) is internal data for downstream subsystems (DQN chord scoring,
    # GPT lyric prompts). The user-relevant facts (key, tempo, notes) are
    # already shown above and in the Story Bible. The raw profile lives
    # behind a collapsed expander as an audit-trail affordance — same
    # mental-model framing as the raw explanation report on the
    # Explanations tab: "advanced/debug, click if curious".
    profile = state.get("melody_profile")
    if profile:
        with st.expander(t("melody.show_profile"), expanded=False):
            st.json(profile)

    latest_result = st.session_state.latest_melody_result
    if latest_result and latest_result.get("session_id") == state["session_id"]:
        st.write(t("melody.latest_extraction"))
        summary_a, summary_b, summary_c = st.columns(3)
        unknown_text = t("melody.unknown")
        summary_a.metric(t("melody.detected_key"), latest_result.get("detected_key") or unknown_text)
        tempo_value = latest_result.get("detected_tempo")
        summary_b.metric(
            t("melody.detected_tempo"),
            f"{tempo_value:.1f} BPM" if tempo_value is not None else unknown_text,
        )
        summary_c.metric(t("melody.returned_notes"), len(latest_result.get("melody_notes", [])))

        # Surface only the *actionable* pipeline signal: a fallback was
        # triggered, so the user should adjust their input. The happy
        # path stays silent — the metrics above already say everything
        # interesting. Contour / harmony explanations were hardcoded
        # placeholder text and have been removed.
        timings = st.session_state.latest_melody_timings
        metadata = (timings or {}).get("metadata", {})
        if metadata.get("used_audio_fallback"):
            st.warning(t("melody.fallback_decode"))
        elif metadata.get("melody_source") == "pesto_rhythm_fallback":
            st.warning(t("melody.fallback_rhythm"))

    if state.get("melody_midi") or state.get("melody_notes"):
        actions_left, actions_right = st.columns(2)
        if actions_left.button(t("melody.play"), use_container_width=True):
            try:
                audio_bytes, source = render_audio_from_session(
                    state,
                    soundfont_path=SOUNDFONT_PATH or None,
                )
                st.session_state.melody_playback_audio = audio_bytes
                st.session_state.melody_playback_source = source
                st.success(t("melody.rendered_via", source=source))
            except Exception as exc:  # pragma: no cover - UI feedback path
                st.error(t("melody.could_not_render", err=exc))

        midi_bytes = decode_midi_bytes(state.get("melody_midi"))
        if midi_bytes:
            actions_right.download_button(
                t("melody.download_midi"),
                data=midi_bytes,
                file_name="melody.mid",
                mime="audio/midi",
                use_container_width=True,
            )

        if st.session_state.melody_playback_audio:
            st.audio(st.session_state.melody_playback_audio, format="audio/wav")
            st.caption(
                t("melody.playback_source", source=st.session_state.melody_playback_source)
            )

    render_melody_continuation_section(state)
    render_refinement_panel("melody")


def render_melody_continuation_section(state: dict[str, Any]) -> None:
    """Melody-continuation generator + variant cards.

    Moved out of the old Suggestions tab and into the Melody tab so the
    "extract → continue" workflow lives in one place. The section labels
    (verse / chorus / etc.) stay English values because the backend
    keys on them; only the display goes through ``t()``.
    """
    st.divider()
    st.markdown(t("suggestions.melody_header"))
    section_options = ["Any", "verse", "pre_chorus", "chorus", "bridge"]
    section_cols = st.columns(2)
    with section_cols[0]:
        primer_section_choice = st.selectbox(
            t("suggestions.primer_label"),
            section_options,
            index=0,
            key=f"primer_section_{state['session_id']}",
            format_func=lambda s: t(f"section.{s}"),
            help=t("suggestions.primer_help"),
        )
    with section_cols[1]:
        target_section_choice = st.selectbox(
            t("suggestions.target_label"),
            section_options,
            index=0,
            key=f"target_section_{state['session_id']}",
            format_func=lambda s: t(f"section.{s}"),
            help=t("suggestions.target_help"),
        )
    primer_section = None if primer_section_choice == "Any" else primer_section_choice
    target_section = None if target_section_choice == "Any" else target_section_choice
    if st.button(
        t("suggestions.generate_melody"),
        use_container_width=True,
        key="generate_melody_continuations",
    ):
        try:
            response = request_melody_suggestions(
                DEFAULT_API_URL,
                session_id=state["session_id"],
                primer_section=primer_section,
                target_section=target_section,
            )
            st.session_state.active_session_state = fetch_session_state(
                DEFAULT_API_URL,
                state["session_id"],
            )
            state = st.session_state.active_session_state
            st.success(t("suggestions.generated_melody_n", n=len(response["melody_suggestions"])))
        except requests.RequestException as exc:
            st.error(t("suggestions.could_not_generate_melody", err=exc))

    melody_suggestions = state.get("melody_suggestions", [])
    if melody_suggestions:

        def render_melody_card(index: int, suggestion: dict[str, Any]) -> None:
            with st.container(border=True):
                st.markdown(t("chords.variant_label", n=index + 1))
                engine = suggestion.get("engine", "mock")
                coherence = float(suggestion["coherence_score"])
                avg_log_prob = suggestion.get("avg_log_prob")
                metric_cols = st.columns(2)
                metric_cols[0].metric(t("suggestions.coherence"), f"{coherence:.2f}")
                if avg_log_prob is not None:
                    metric_cols[1].metric(t("suggestions.avg_log_p"), f"{float(avg_log_prob):.2f}")
                else:
                    metric_cols[1].metric(t("suggestions.engine"), engine)
                st.caption(suggestion["explanation"])
                if suggestion.get("notes"):
                    try:
                        st.audio(
                            synthesize_wave_from_notes(
                                suggestion["notes"],
                                bpm=float(state.get("detected_tempo") or 100.0),
                            ),
                            format="audio/wav",
                        )
                    except Exception:
                        st.caption(t("suggestions.preview_unavailable"))
                if st.button(
                    t("suggestions.use_continuation"),
                    key=f"accept_melody_suggestion_{index + 1}",
                    use_container_width=True,
                ):
                    try:
                        response = accept_melody_suggestion(
                            DEFAULT_API_URL,
                            session_id=state["session_id"],
                            suggestion_index=index,
                        )
                        st.session_state.active_session_state = response["state"]
                        push_snapshot("snapshot.accepted_continuation", n=index + 1)
                        st.success(t("suggestions.melody_extended"))
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(t("suggestions.could_not_apply", err=exc))

        render_grid(
            melody_suggestions,
            cols_per_row=3,
            render_card=render_melody_card,
        )
    else:
        st.caption(t("suggestions.no_continuations"))


def render_chords_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader(t("chords.subheader"))
    if not state:
        st.write(t("chords.context_placeholder"))
        return

    # Two parallel chord-generation sources — pick whichever the
    # session currently has material for. The buttons are gated on the
    # required state so the user can't trigger a 422 (e.g. melody-DQN
    # without an uploaded melody, or lyrics-DQN without lyrics).
    lyrics_in_state = (state.get("lyrics_text") or "").strip()
    melody_notes = state.get("melody_notes") or []
    has_melody = len(melody_notes) > 0

    lyrics_col, melody_col = st.columns(2)

    with lyrics_col:
        st.markdown(t("chords.from_lyrics_header"))
        if lyrics_in_state:
            with st.container(border=True):
                st.write(lyrics_in_state)
        else:
            st.info(t("chords.lyrics_empty_hint"))

        if st.button(
            t("chords.from_lyrics_button"),
            use_container_width=True,
            disabled=not lyrics_in_state,
            key="chords_from_lyrics_submit",
        ):
            try:
                response = generate_chords_from_lyrics(
                    DEFAULT_API_URL,
                    session_id=state["session_id"],
                    lyrics_text=lyrics_in_state,
                )
                st.session_state.active_session_state = fetch_session_state(
                    DEFAULT_API_URL,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                push_snapshot(
                    "snapshot.generated_chords",
                    n=len(response["chord_progressions"]),
                    mood=response["mood"],
                )
                st.success(
                    t(
                        "chords.generated_n",
                        n=len(response["chord_progressions"]),
                        mood=response["mood"],
                    )
                )
                # Suggest-not-override: if the author's Song Brief mood was kept,
                # stash the lyric-inferred mood so the affordance below can offer
                # to adopt it. Survives the rerun via session_state.
                if response.get("mood_suggestion_pending") and response.get("detected_mood"):
                    st.session_state["pending_detected_mood"] = response["detected_mood"]
                else:
                    st.session_state.pop("pending_detected_mood", None)
            except requests.RequestException as exc:
                st.error(t("chords.could_not_generate", err=exc))

        pending_mood = st.session_state.get("pending_detected_mood")
        if pending_mood and normalize_mood(pending_mood):
            canonical = normalize_mood(pending_mood)
            st.info(t("chords.mood_suggested", mood=t(f"mood.{canonical}")))
            if st.button(
                t("chords.accept_detected_mood"),
                key="accept_detected_mood",
                use_container_width=True,
            ):
                try:
                    set_session_mood(
                        DEFAULT_API_URL,
                        session_id=state["session_id"],
                        mood=canonical,
                    )
                    st.session_state.song_brief["mood"] = canonical
                    st.session_state.active_session_state = fetch_session_state(
                        DEFAULT_API_URL, state["session_id"]
                    )
                    st.session_state.pop("pending_detected_mood", None)
                    push_snapshot("snapshot.mood_set", mood=t(f"mood.{canonical}"))
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(t("brief.could_not_set_mood", err=exc))

    with melody_col:
        st.markdown(t("chords.from_melody_header"))
        if has_melody:
            detected_key = state.get("detected_key") or t("melody.unknown")
            st.caption(
                t(
                    "chords.melody_notes_preview",
                    n=len(melody_notes),
                    key=detected_key,
                )
            )
        else:
            st.info(t("chords.melody_empty_hint"))

        if st.button(
            t("chords.from_melody_button"),
            use_container_width=True,
            disabled=not has_melody,
            key="chords_from_melody_submit",
        ):
            try:
                response = generate_chords_from_melody(
                    DEFAULT_API_URL,
                    session_id=state["session_id"],
                )
                st.session_state.active_session_state = fetch_session_state(
                    DEFAULT_API_URL,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                push_snapshot(
                    "snapshot.generated_chords_from_melody",
                    n=len(response["chord_progressions"]),
                )
                st.success(
                    t(
                        "chords.generated_from_melody_n",
                        n=len(response["chord_progressions"]),
                    )
                )
            except requests.RequestException as exc:
                st.error(t("chords.could_not_generate_from_melody", err=exc))

    st.markdown("---")
    st.markdown(t("chords.manual_header"))
    st.caption(t("chords.manual_caption"))
    manual_chord_input = st.text_input(
        t("chords.progression_label"),
        value=st.session_state.get("manual_chord_input", ""),
        key="manual_chord_input",
        placeholder=t("chords.progression_placeholder"),
    )
    parsed_chords = parse_manual_chord_input(manual_chord_input)
    if manual_chord_input and parsed_chords:
        st.caption(t("chords.parsed", chords=" -> ".join(parsed_chords)))
    if st.button(
        t("chords.use_progression"),
        use_container_width=True,
        disabled=not parsed_chords,
        key="manual_chord_submit",
    ):
        try:
            submit_manual_chords(
                DEFAULT_API_URL,
                session_id=state["session_id"],
                chords=parsed_chords,
            )
            st.session_state.active_session_state = fetch_session_state(
                DEFAULT_API_URL,
                state["session_id"],
            )
            state = st.session_state.active_session_state
            push_snapshot(
                "snapshot.manual_progression",
                chords=" -> ".join(parsed_chords),
            )
            st.success(t("chords.saved_manual", n=len(parsed_chords)))
        except requests.RequestException as exc:
            st.error(t("chords.could_not_save", err=exc))

    progressions = state.get("chord_progressions", [])
    if progressions:
        st.markdown(t("chords.candidate_header"))
        st.caption(t("chords.candidate_caption"))

        def render_chord_card(index: int, progression: dict[str, Any]) -> None:
            with st.container(border=True):
                st.markdown(t("chords.variant_label", n=index + 1))
                st.markdown(progression_title(progression))

                # Audio preview — synthesize the progression so the user
                # can actually hear it. Consecutive duplicate chords are
                # collapsed at the audio level too (a "Caug ×18" card
                # plays one Caug, not 40 seconds of the same chord).
                # Tempo follows the detected BPM so the preview lines up
                # with what the melody would sound at.
                chords_for_audio = progression.get("chords") or []
                if chords_for_audio:
                    try:
                        preview_bpm = float(state.get("detected_tempo") or 100.0)
                        st.audio(
                            synthesize_progression_audio(
                                chords_for_audio,
                                bpm=preview_bpm,
                            ),
                            format="audio/wav",
                        )
                    except Exception:
                        # Silent degrade: unknown chord symbol, empty
                        # sequence, etc. The card stays usable; the user
                        # just doesn't get a preview for this variant.
                        pass

                # Score badges — same st.metric pattern as melody cards.
                score_cols = st.columns(2)
                model_confidence = progression.get(
                    "model_confidence", progression.get("score")
                )
                mood_alignment = progression.get("mood_alignment")
                if model_confidence is not None:
                    score_cols[0].metric(
                        t("chords.model_confidence"),
                        f"{float(model_confidence):.2f}",
                    )
                if mood_alignment is not None:
                    score_cols[1].metric(
                        t("chords.mood_alignment"),
                        f"{float(mood_alignment):.2f}",
                    )
                harmonic_function = progression.get("harmonic_function")
                if harmonic_function:
                    st.caption(
                        f"**{t('chords.harmonic_function')}:** {harmonic_function}"
                    )

                # All technical detail under one collapsed expander —
                # consistent with the "Show melody profile (advanced)"
                # and "Show raw report (advanced)" pattern elsewhere.
                explanation_text = (progression.get("explanation") or "").strip()
                annotations = progression.get("chord_annotations") or []
                distributions = progression.get("native_distributions") or []
                if explanation_text or annotations or distributions:
                    with st.expander(t("chords.show_technical"), expanded=False):
                        if explanation_text:
                            st.caption(explanation_text)
                        render_progression_details(progression)

                if st.button(
                    t("chords.use_progression"),
                    key=f"accept_progression_{index}",
                    use_container_width=True,
                ):
                    try:
                        submit_manual_chords(
                            DEFAULT_API_URL,
                            session_id=state["session_id"],
                            chords=list(progression["chords"]),
                        )
                        st.session_state.active_session_state = fetch_session_state(
                            DEFAULT_API_URL,
                            state["session_id"],
                        )
                        push_snapshot(
                            "snapshot.selected_progression",
                            n=index + 1,
                            title=progression_title(progression),
                        )
                        st.success(t("chords.committed_variant", n=index + 1))
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(t("chords.could_not_commit", err=exc))

        render_grid(progressions, cols_per_row=4, render_card=render_chord_card)
    else:
        st.caption(t("chords.no_progressions"))

    render_refinement_panel("chords")


def render_lyrics_tab() -> None:
    """Lyrics tab: typed lyrics buffer + lyric suggestion generator.

    Owns the lyrics text-input that the Chords tab and the GPT lyric
    prompts both consume. Clicking Generate POSTs the buffer to
    ``/suggest/lyrics`` (which persists ``state.lyrics_text`` before
    generating), then renders the returned variants as a grid.
    """
    state = st.session_state.active_session_state
    st.subheader(t("lyrics.subheader"))
    if not state:
        st.write(t("lyrics.placeholder"))
        return

    # The buffer is pre-seeded from `state.lyrics_text` on session
    # create/load/restore (see `reset_lyrics_input`); the widget's `key`
    # keeps the user's edits sticky across reruns. We must NOT pass
    # `value=` here — Streamlit ignores it once the key exists and would
    # emit a double-source warning.
    lyrics_text = st.text_area(
        t("lyrics.input_label"),
        key="lyrics_input",
        height=200,
        placeholder=t("lyrics.input_placeholder"),
        help=t("lyrics.input_help"),
    )

    # Style picker — closed taxonomy matching backend's VALID_LYRIC_MODES.
    # Values stay English (backend keys on them); only the display goes
    # through t(). Defaulting to Poetic matches the backend's silent
    # fallback so the visible selection always agrees with what runs.
    mode_col, _ = st.columns([1, 2])
    selected_mode = mode_col.selectbox(
        t("lyrics.mode_label"),
        options=LYRIC_MODE_OPTIONS,
        index=LYRIC_MODE_OPTIONS.index(LYRIC_MODE_DEFAULT),
        key="lyric_mode",
        format_func=lambda m: t(f"lyric_mode.{m}"),
        help=t("lyrics.mode_help"),
    )

    if st.button(
        t("lyrics.generate_button"),
        use_container_width=True,
        key="generate_lyric_suggestions",
    ):
        try:
            response = request_lyric_suggestions(
                DEFAULT_API_URL,
                session_id=state["session_id"],
                lyrics_text=lyrics_text,
                mode=selected_mode,
            )
            st.session_state.active_session_state = fetch_session_state(
                DEFAULT_API_URL,
                state["session_id"],
            )
            state = st.session_state.active_session_state
            push_snapshot(
                "snapshot.generated_lyrics",
                n=len(response["lyric_suggestions"]),
            )
            st.success(t("lyrics.generated_n", n=len(response["lyric_suggestions"])))
        except requests.RequestException as exc:
            st.error(t("lyrics.could_not_generate", err=exc))

    lyric_suggestions = state.get("lyric_suggestions", [])
    if lyric_suggestions:

        def render_lyric_card(index: int, suggestion: dict[str, Any]) -> None:
            with st.container(border=True):
                st.markdown(t("chords.variant_label", n=index + 1))
                st.write(suggestion["text"])
                # Translate the mode value through the same dict the
                # dropdown uses, so the caption matches the picker
                # regardless of language. Fall back to the raw value if
                # the model returned an unknown mode (defensive — the
                # caption stays readable rather than printing a key).
                raw_mode = str(suggestion.get("mode") or LYRIC_MODE_DEFAULT)
                mode_key = f"lyric_mode.{raw_mode}"
                mode_display = t(mode_key) if mode_key != t(mode_key) else raw_mode
                st.caption(
                    t(
                        "suggestions.lyric_caption",
                        mode=mode_display,
                        syllables=suggestion["syllable_count"],
                    )
                )

        render_grid(
            lyric_suggestions,
            cols_per_row=3,
            render_card=render_lyric_card,
        )
    else:
        st.caption(t("lyrics.no_suggestions"))

    render_refinement_panel("lyrics")


def render_explanations_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader(t("explanations.subheader"))
    if not state:
        st.write(t("explanations.placeholder"))
        return

    # Chat on the left, structured report on the right — chat is the
    # active user surface, the report is reference material.
    chat_column, report_column = st.columns(2)
    explanation_report = state.get("explanation_report")

    with report_column:
        st.markdown(t("explanations.xai_header"))
        if explanation_report:
            # Friendly summary as primary content. The structured fields
            # under it (constraint_logs, melody_confidence, cache_status …)
            # are diagnostic data, not user-facing copy — they belong
            # behind an audit-trail expander, not in the main view.
            summary = (explanation_report.get("summary") or "").strip()
            if summary:
                st.write(summary)

            # Provenance line: what just ran + which model + cache state.
            # Built defensively so missing pieces just shrink the line
            # rather than blanking it.
            badges: list[str] = []
            source_action = explanation_report.get("source_action")
            if source_action:
                badges.append(
                    f"**{t('explanations.action_label')}:** "
                    f"{_friendly_action_label(str(source_action))}"
                )
            cache_status = explanation_report.get("cache_status") or {}
            use_case = cache_status.get("use_case")
            if use_case:
                model = cache_status.get(f"last_{use_case}_model_version")
                cache_hit = cache_status.get(f"last_{use_case}_cache_hit")
                if model:
                    badges.append(
                        f"**{t('explanations.model_label')}:** "
                        f"{_short_model_label(str(model))}"
                    )
                if cache_hit is not None:
                    cache_label = (
                        t("explanations.cache_hit")
                        if cache_hit
                        else t("explanations.cache_miss")
                    )
                    badges.append(
                        f"**{t('explanations.cache_label')}:** {cache_label}"
                    )
            if badges:
                st.caption(" · ".join(badges))

            # Raw report kept available for thesis-defense moments
            # ("show me the audit trail"). Collapsed by default — the
            # expander reads as "advanced/debug", which is the right
            # mental model for internal structured data.
            with st.expander(t("explanations.show_raw_report"), expanded=False):
                st.json(explanation_report)
        else:
            st.caption(t("explanations.no_report"))

    with chat_column:
        st.markdown(t("explanations.chat_header"))
        chat_history = state.get("chat_history", [])
        if chat_history:
            you_label = t("explanations.you")
            bot_label = t("explanations.bot")
            for message in chat_history[-10:]:
                label = you_label if message["role"] == "user" else bot_label
                st.markdown(f"**{label}:** {message['content']}")
        else:
            st.caption(t("explanations.placeholder_ask"))

        # Streamlit forbids writing to a widget-bound key after the widget
        # is instantiated, so apply any pending clear *before* creating it.
        if st.session_state.pop("_clear_explanations_chat_text", False):
            st.session_state["explanations_chat_text"] = ""

        prompt = st.text_input(
            t("explanations.input_label"),
            key="explanations_chat_text",
            placeholder=t("explanations.input_placeholder"),
        )
        if st.button(t("explanations.send_button"), use_container_width=True, disabled=not prompt.strip()):
            try:
                response = send_chat_message(
                    DEFAULT_API_URL,
                    session_id=state["session_id"],
                    message=prompt.strip(),
                    language=LANGUAGES.get(
                        st.session_state.get("language", DEFAULT_LANGUAGE),
                        LANGUAGES[DEFAULT_LANGUAGE],
                    ),
                )
                st.session_state.active_session_state = fetch_session_state(
                    DEFAULT_API_URL,
                    state["session_id"],
                )
                st.session_state["_clear_explanations_chat_text"] = True
                st.success(t("explanations.replied"))
                st.rerun()
            except requests.RequestException as exc:
                st.error(t("explanations.could_not_send", err=exc))


def render_refinement_panel(target: str) -> None:
    state = st.session_state.active_session_state
    if not state:
        return

    key = f"{target}_refine_text"
    st.divider()
    st.markdown(t("refine.header"))
    instruction = st.text_input(
        t("refine.placeholder"),
        key=key,
        placeholder=t("refine.example_placeholder"),
        label_visibility="collapsed",
    )
    target_label = t(f"target.{target}")
    if st.button(
        t(f"refine.button.{target}"),
        key=f"refine_{target}",
        use_container_width=True,
        disabled=not instruction.strip(),
    ):
        try:
            response = refine_session(
                DEFAULT_API_URL,
                session_id=state["session_id"],
                instruction=instruction.strip(),
                target=target,
            )
            st.session_state.active_session_state = response["state"]
            trimmed = instruction.strip()
            if len(trimmed) > 40:
                trimmed = trimmed[:37] + "..."
            push_snapshot(
                "snapshot.refined",
                target=target_label,
                instruction=trimmed,
            )
            st.success(t("refine.applied", target=target_label))
        except requests.RequestException as exc:
            st.error(t("refine.could_not", target=target_label, err=exc))
    interpretation = state.get("user_params", {}).get("last_refinement_interpretation")
    if interpretation and state.get("user_params", {}).get("last_refinement_target") == target:
        st.caption(t("refine.interpretation", text=interpretation))


def main() -> None:
    ensure_session_defaults()
    render_header()
    render_sidebar()

    if st.session_state.active_session_id and st.button(t("main.refresh_active")):
        try:
            sync_active_session(DEFAULT_API_URL)
            st.success(t("main.refreshed"))
        except requests.RequestException as exc:
            st.error(t("main.could_not_refresh", err=exc))

    render_session_overview()
    render_song_brief()
    melody_tab, chords_tab, lyrics_tab, explanations_tab = st.tabs(
        [t("tab.melody"), t("tab.chords"), t("tab.lyrics"), t("tab.explanations")]
    )
    with melody_tab:
        render_melody_tab()
    with chords_tab:
        render_chords_tab()
    with lyrics_tab:
        render_lyrics_tab()
    with explanations_tab:
        render_explanations_tab()


if __name__ == "__main__":
    main()
