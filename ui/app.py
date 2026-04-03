"""Streamlit UI shell for the HumMuse co-creative workflow."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

try:
    from ui.audio_utils import render_audio_from_session
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from audio_utils import render_audio_from_session


DEFAULT_API_URL = "http://127.0.0.1:8000"
SUPPORTED_AUDIO_TYPES = ["wav", "webm", "mp3"]


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
    prompt: str,
    mood: str,
    tempo_bpm: int,
) -> dict[str, Any]:
    audio_file = build_audio_file_tuple(uploaded_audio)
    return api_post_multipart(
        api_base_url,
        "/melody/from-hum",
        files={"audio": audio_file},
        data={
            "session_id": session_id,
            "prompt": prompt,
            "mood": mood,
            "tempo_bpm": str(tempo_bpm),
        },
    )


def note_rows(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pitch": note["pitch"],
            "onset": note["onset"],
            "duration": note["duration"],
            "velocity": note["velocity"],
            "confidence": note["confidence"],
        }
        for note in notes
    ]


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


def recognise_chords_from_melody(
    api_base_url: str,
    *,
    session_id: str,
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/chords/from-melody",
        payload={"session_id": session_id},
    )


def request_melody_suggestions(
    api_base_url: str,
    *,
    session_id: str,
    num_suggestions: int = 3,
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/melody/continue",
        payload={"session_id": session_id, "num_suggestions": num_suggestions},
    )


def request_lyric_suggestions(
    api_base_url: str,
    *,
    session_id: str,
    mode: str = "continue",
    num_suggestions: int = 3,
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        "/suggest/lyrics",
        payload={"session_id": session_id, "mode": mode, "num_suggestions": num_suggestions},
    )


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
) -> dict[str, Any]:
    return api_post(
        api_base_url,
        f"/session/{session_id}/chat",
        payload={"message": message},
    )


def progression_title(progression: dict[str, Any]) -> str:
    return " -> ".join(progression["chords"])


def progression_caption(progression: dict[str, Any]) -> str:
    score = progression.get("score")
    explanation = progression.get("explanation", "")
    harmonic_function = progression.get("harmonic_function")
    if harmonic_function:
        explanation = f"{harmonic_function} | {explanation}"
    if score is None:
        return explanation
    return f"Score: {score:.2f} | {explanation}"


def numbered_label(index: int, text: str) -> str:
    return f"{index}. {text}"


def confidence_to_color(confidence: float) -> str:
    if confidence >= 0.85:
        return "#d9f99d"
    if confidence >= 0.7:
        return "#fde68a"
    return "#fecaca"


def render_confidence_note_table(notes: list[dict[str, Any]]) -> None:
    rows = [
        "<tr>"
        f"<td>{note['pitch']}</td>"
        f"<td>{note['onset']:.2f}</td>"
        f"<td>{note['duration']:.2f}</td>"
        f"<td>{note['velocity']}</td>"
        f"<td style='background:{confidence_to_color(float(note['confidence']))};'>{float(note['confidence']):.2f}</td>"
        "</tr>"
        for note in notes
    ]
    st.markdown(
        """
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr>
              <th style="text-align:left;">Pitch</th>
              <th style="text-align:left;">Onset</th>
              <th style="text-align:left;">Duration</th>
              <th style="text-align:left;">Velocity</th>
              <th style="text-align:left;">Confidence</th>
            </tr>
          </thead>
          <tbody>
        """
        + "".join(rows)
        + """
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def format_session_option(session: dict[str, Any]) -> str:
    session_id = session["session_id"]
    updated_at = session["updated_at"].replace("T", " ").replace("Z", " UTC")
    return f"{session_id} | updated {updated_at}"


def ensure_session_defaults() -> None:
    st.session_state.setdefault("api_base_url", DEFAULT_API_URL)
    st.session_state.setdefault("sessions_cache", [])
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("active_session_state", None)
    st.session_state.setdefault("selected_session_label", None)
    st.session_state.setdefault("melody_prompt", "Melody sketch from humming")
    st.session_state.setdefault("melody_mood", "uplift")
    st.session_state.setdefault("melody_tempo", 120)
    st.session_state.setdefault("lyrics_input", "")
    st.session_state.setdefault("melody_refine_text", "")
    st.session_state.setdefault("chords_refine_text", "")
    st.session_state.setdefault("suggestions_refine_text", "")
    st.session_state.setdefault("explanations_chat_text", "")
    st.session_state.setdefault("soundfont_path", os.environ.get("HUMMUSE_SOUNDFONT", ""))
    st.session_state.setdefault("melody_playback_audio", None)
    st.session_state.setdefault("melody_playback_source", None)


def clear_playback_cache() -> None:
    st.session_state.melody_playback_audio = None
    st.session_state.melody_playback_source = None


def refresh_sessions(api_base_url: str) -> None:
    st.session_state.sessions_cache = load_sessions(api_base_url)


def sync_active_session(api_base_url: str) -> None:
    active_session_id = st.session_state.active_session_id
    if active_session_id:
        st.session_state.active_session_state = fetch_session_state(api_base_url, active_session_id)
        clear_playback_cache()


def render_sidebar() -> None:
    st.sidebar.title("HumMuse")
    st.sidebar.caption("Session-first songwriting workspace")

    api_base_url = st.sidebar.text_input("API base URL", key="api_base_url")
    st.sidebar.text_input("SoundFont Path (Optional)", key="soundfont_path")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("New Session", use_container_width=True):
        try:
            created = create_session(api_base_url)
            state = created["state"]
            st.session_state.active_session_id = state["session_id"]
            st.session_state.active_session_state = state
            clear_playback_cache()
            refresh_sessions(api_base_url)
            st.session_state.selected_session_label = next(
                (
                    format_session_option(session)
                    for session in st.session_state.sessions_cache
                    if session["session_id"] == state["session_id"]
                ),
                None,
            )
            st.sidebar.success(f"Created session {state['session_id']}")
        except requests.RequestException as exc:
            st.sidebar.error(f"Could not create session: {exc}")

    if col_b.button("Refresh", use_container_width=True):
        try:
            refresh_sessions(api_base_url)
            st.sidebar.success("Session list refreshed")
        except requests.RequestException as exc:
            st.sidebar.error(f"Could not load sessions: {exc}")

    if not st.session_state.sessions_cache:
        try:
            refresh_sessions(api_base_url)
        except requests.RequestException:
            pass

    options = {format_session_option(session): session["session_id"] for session in st.session_state.sessions_cache}
    selected_label = st.sidebar.selectbox(
        "Load Existing Session",
        options=["None", *options.keys()],
        index=0 if st.session_state.selected_session_label not in options else list(options.keys()).index(st.session_state.selected_session_label) + 1,
    )

    if st.sidebar.button("Load Session", use_container_width=True):
        if selected_label == "None":
            st.sidebar.info("Choose a session from the list first.")
        else:
            session_id = options[selected_label]
            try:
                st.session_state.active_session_state = fetch_session_state(api_base_url, session_id)
                st.session_state.active_session_id = session_id
                st.session_state.selected_session_label = selected_label
                clear_playback_cache()
                st.sidebar.success(f"Loaded session {session_id}")
            except requests.RequestException as exc:
                st.sidebar.error(f"Could not load session: {exc}")

    st.sidebar.divider()
    active_session_id = st.session_state.active_session_id
    if active_session_id:
        st.sidebar.markdown("**Active Session**")
        st.sidebar.code(active_session_id)
    else:
        st.sidebar.info("Create or load a session to begin.")


def render_header() -> None:
    st.set_page_config(page_title="HumMuse", page_icon="🎼", layout="wide")
    st.title("HumMuse")
    st.caption("AI co-creative songwriting assistant")


def render_session_overview() -> None:
    state = st.session_state.active_session_state
    if not state:
        st.info("Use the sidebar to create a session or load an existing one.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Session", state["session_id"][:8])
    col_b.metric("Melody Notes", len(state.get("melody_notes", [])))
    col_c.metric("Lyric Suggestions", len(state.get("lyric_suggestions", [])))
    col_d.metric("Chat Turns", len(state.get("chat_history", [])))

    st.caption(f"Current session id: `{state['session_id']}`")
    last_refinement = state.get("user_params", {}).get("last_refinement")
    if last_refinement:
        target = state.get("user_params", {}).get("last_refinement_target", "session")
        st.caption(f"Latest refinement for `{target}`: {last_refinement}")
    interpretation = state.get("user_params", {}).get("last_refinement_interpretation")
    if interpretation:
        st.caption(f"Interpretation: {interpretation}")


def render_melody_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader("Melody")
    if not state:
        st.write("Session context will appear here after you create or load one.")
        return

    st.write("Upload or record a melody sketch")
    recorded_audio = st.audio_input("Record humming")
    uploaded_audio = st.file_uploader(
        "Or upload an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        help="Accepted formats: WAV, WebM, MP3",
    )

    chosen_audio = recorded_audio or uploaded_audio
    if chosen_audio is not None:
        _, audio_bytes, mime_type = build_audio_file_tuple(chosen_audio)
        st.audio(audio_bytes, format=mime_type)

    controls_a, controls_b, controls_c = st.columns(3)
    controls_a.text_input("Prompt", key="melody_prompt")
    controls_b.text_input("Mood", key="melody_mood")
    controls_c.number_input("Tempo BPM", min_value=40, max_value=240, key="melody_tempo")

    if st.button("Extract Melody", use_container_width=True, disabled=chosen_audio is None):
        try:
            response = extract_melody_from_audio(
                st.session_state.api_base_url,
                session_id=state["session_id"],
                uploaded_audio=chosen_audio,
                prompt=st.session_state.melody_prompt,
                mood=st.session_state.melody_mood,
                tempo_bpm=int(st.session_state.melody_tempo),
            )
            st.session_state.active_session_state = fetch_session_state(
                st.session_state.api_base_url,
                response["session_id"],
            )
            clear_playback_cache()
            state = st.session_state.active_session_state
            st.success(f"Extracted {len(response['melody_notes'])} note events.")
        except requests.RequestException as exc:
            st.error(f"Could not extract melody: {exc}")

    notes = state.get("melody_notes", [])
    if notes:
        st.write("Current extracted notes")
        st.caption("Confidence colors: green = strong, amber = usable, red = low confidence.")
        render_confidence_note_table(note_rows(notes))
    else:
        st.write("No melody extracted yet. Record or upload audio, then click `Extract Melody`.")

    profile = state.get("melody_profile")
    if profile:
        st.write("Melody profile")
        st.json(profile)

    if state.get("melody_midi") or state.get("melody_notes"):
        if st.button("Play Melody", use_container_width=True):
            try:
                audio_bytes, source = render_audio_from_session(
                    state,
                    soundfont_path=st.session_state.soundfont_path.strip() or None,
                )
                st.session_state.melody_playback_audio = audio_bytes
                st.session_state.melody_playback_source = source
                st.success(f"Rendered playback using {source}.")
            except Exception as exc:  # pragma: no cover - UI feedback path
                st.error(f"Could not render playback: {exc}")

        if st.session_state.melody_playback_audio:
            st.audio(st.session_state.melody_playback_audio, format="audio/wav")
            st.caption(
                f"Playback source: `{st.session_state.melody_playback_source}`"
            )

    render_refinement_panel("melody")


def render_chords_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader("Chords")
    if not state:
        st.write("Chord generation will use the active session once selected.")
        return

    melody_column, lyric_column = st.columns(2)

    with melody_column:
        st.markdown("**Recognise from melody**")
        st.caption("Use the current melody notes in the active session to produce mock recognised chords.")
        if st.button(
            "Recognise Chords From Melody",
            use_container_width=True,
            disabled=not state.get("melody_notes"),
        ):
            try:
                response = recognise_chords_from_melody(
                    st.session_state.api_base_url,
                    session_id=state["session_id"],
                )
                st.session_state.active_session_state = fetch_session_state(
                    st.session_state.api_base_url,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                st.success(f"Recognised {len(response['recognised_chords'])} chord spans from the melody.")
            except requests.RequestException as exc:
                st.error(f"Could not recognise chords from melody: {exc}")

        recognised_chords = state.get("recognised_chords", [])
        if recognised_chords:
            for chord in recognised_chords:
                with st.container(border=True):
                    st.markdown(f"**{chord['symbol']}**")
                    st.caption(
                        f"Beats {chord['start_beat']:.1f}-{chord['start_beat'] + chord['duration_beats']:.1f} | "
                        f"Confidence: {chord['confidence']:.2f} | {chord.get('harmonic_function') or 'function pending'}"
                    )
                    if chord.get("decoding_trace"):
                        with st.expander("Decoding trace"):
                            st.json(chord["decoding_trace"])
        else:
            st.caption("No recognised chords yet.")

    with lyric_column:
        st.markdown("**Generate from lyrics**")
        lyrics_default = state.get("lyrics_text") or st.session_state.lyrics_input
        lyrics_text = st.text_area(
            "Lyrics",
            value=lyrics_default,
            key="lyrics_input",
            height=160,
            placeholder="Type or paste your lyrics here, then generate candidate progressions.",
        )

        if st.button("Generate Chords", use_container_width=True, disabled=not lyrics_text.strip()):
            try:
                response = generate_chords_from_lyrics(
                    st.session_state.api_base_url,
                    session_id=state["session_id"],
                    lyrics_text=lyrics_text.strip(),
                )
                st.session_state.active_session_state = fetch_session_state(
                    st.session_state.api_base_url,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                st.success(
                    f"Generated {len(response['chord_progressions'])} candidate progressions for a {response['mood']} mood."
                )
            except requests.RequestException as exc:
                st.error(f"Could not generate chords: {exc}")

        progressions = state.get("chord_progressions", [])
        if progressions:
            for progression in progressions:
                with st.container(border=True):
                    st.markdown(f"**{progression_title(progression)}**")
                    st.caption(progression_caption(progression))
        else:
            st.caption("No lyric-driven chord progressions yet.")

    render_refinement_panel("chords")


def render_suggestions_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader("Suggestions")
    if not state:
        st.write("Suggestion tools become available once a session is active.")
        return

    melody_column, lyric_column = st.columns(2)

    with melody_column:
        st.markdown("**Melody continuation**")
        if st.button("Generate Melody Continuations", use_container_width=True):
            try:
                response = request_melody_suggestions(
                    st.session_state.api_base_url,
                    session_id=state["session_id"],
                )
                st.session_state.active_session_state = fetch_session_state(
                    st.session_state.api_base_url,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                st.success(f"Generated {len(response['melody_suggestions'])} melody options.")
            except requests.RequestException as exc:
                st.error(f"Could not generate melody continuations: {exc}")

        melody_suggestions = state.get("melody_suggestions", [])
        if melody_suggestions:
            for index, suggestion in enumerate(melody_suggestions, start=1):
                st.write(numbered_label(index, suggestion["explanation"]))
                st.caption(f"Coherence score: {suggestion['coherence_score']:.2f}")
        else:
            st.caption("No melody continuations yet.")

    with lyric_column:
        st.markdown("**Lyric suggestions**")
        if st.button("Generate Lyric Suggestions", use_container_width=True):
            try:
                response = request_lyric_suggestions(
                    st.session_state.api_base_url,
                    session_id=state["session_id"],
                )
                st.session_state.active_session_state = fetch_session_state(
                    st.session_state.api_base_url,
                    state["session_id"],
                )
                state = st.session_state.active_session_state
                st.success(f"Generated {len(response['lyric_suggestions'])} lyric options.")
            except requests.RequestException as exc:
                st.error(f"Could not generate lyric suggestions: {exc}")

        lyric_suggestions = state.get("lyric_suggestions", [])
        if lyric_suggestions:
            for index, suggestion in enumerate(lyric_suggestions, start=1):
                st.write(numbered_label(index, suggestion["text"]))
                st.caption(
                    f"Mode: {suggestion['mode']} | Syllables: {suggestion['syllable_count']}"
                )
        else:
            st.caption("No lyric suggestions yet.")

    render_refinement_panel("suggestions")


def render_explanations_tab() -> None:
    state = st.session_state.active_session_state
    st.subheader("Explanations")
    if not state:
        st.write("Explanation tools become available once a session is active.")
        return

    report_column, chat_column = st.columns(2)
    explanation_report = state.get("explanation_report")

    with report_column:
        st.markdown("**Structured XAI data**")
        if explanation_report:
            st.caption(explanation_report.get("summary", ""))
            st.json(explanation_report)
        else:
            st.caption("No explanation report yet. Run a melody, chord, or suggestion action first.")

    with chat_column:
        st.markdown("**Chat panel**")
        chat_history = state.get("chat_history", [])
        if chat_history:
            for message in chat_history[-10:]:
                label = "You" if message["role"] == "user" else "HumMuse"
                st.markdown(f"**{label}:** {message['content']}")
        else:
            st.caption("Ask why the system chose something after generating some material.")

        prompt = st.text_input(
            "Ask about the current explanation report",
            key="explanations_chat_text",
            placeholder="Why did you choose Dm here?",
        )
        if st.button("Send Question", use_container_width=True, disabled=not prompt.strip()):
            try:
                response = send_chat_message(
                    st.session_state.api_base_url,
                    session_id=state["session_id"],
                    message=prompt.strip(),
                )
                st.session_state.active_session_state = fetch_session_state(
                    st.session_state.api_base_url,
                    state["session_id"],
                )
                st.session_state.explanations_chat_text = ""
                st.success("Received explanation reply.")
            except requests.RequestException as exc:
                st.error(f"Could not send explanation question: {exc}")


def render_refinement_panel(target: str) -> None:
    state = st.session_state.active_session_state
    if not state:
        return

    key = f"{target}_refine_text"
    st.divider()
    st.markdown("**Refine This Section**")
    instruction = st.text_input(
        "Refinement instruction",
        key=key,
        placeholder="Example: make it jazzier",
        label_visibility="collapsed",
    )
    if st.button(f"Refine {target.title()}", key=f"refine_{target}", use_container_width=True, disabled=not instruction.strip()):
        try:
            response = refine_session(
                st.session_state.api_base_url,
                session_id=state["session_id"],
                instruction=instruction.strip(),
                target=target,
            )
            st.session_state.active_session_state = response["state"]
            st.success(f"Applied refinement to {target}.")
        except requests.RequestException as exc:
            st.error(f"Could not refine {target}: {exc}")
    interpretation = state.get("user_params", {}).get("last_refinement_interpretation")
    if interpretation and state.get("user_params", {}).get("last_refinement_target") == target:
        st.caption(f"Interpretation: {interpretation}")


def main() -> None:
    ensure_session_defaults()
    render_header()
    render_sidebar()

    if st.session_state.active_session_id and st.button("Refresh Active Session"):
        try:
            sync_active_session(st.session_state.api_base_url)
            st.success("Session refreshed")
        except requests.RequestException as exc:
            st.error(f"Could not refresh session: {exc}")

    render_session_overview()
    melody_tab, chords_tab, suggestions_tab, explanations_tab = st.tabs(
        ["Melody", "Chords", "Suggestions", "Explanations"]
    )
    with melody_tab:
        render_melody_tab()
    with chords_tab:
        render_chords_tab()
    with suggestions_tab:
        render_suggestions_tab()
    with explanations_tab:
        render_explanations_tab()


if __name__ == "__main__":
    main()
