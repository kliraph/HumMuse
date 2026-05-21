"""Internationalization for the HumMuse Streamlit UI.

Two languages today: English (default) and Russian. Strings are keyed
by stable, language-independent identifiers (dotted paths like
``sidebar.new_session``). The current language lives in
``st.session_state.language``; flipping it triggers a rerun and every
``t(key)`` call returns the new string on the next pass.

Conventions kept by callers in ``ui/app.py``:

- Widget ``key=`` arguments stay English. Streamlit identifies widgets
  by ``key``, so translating those would reset widget state on every
  toggle. Only the visible label/help/placeholder/caption strings go
  through ``t``.
- Snapshot labels are stored as (key, kwargs) tuples on the snapshot
  object and resolved at render time. That way version-history entries
  follow the active language even after the snapshot was taken in
  another one.
- Backend-returned strings (``progression['explanation']``,
  ``last_refinement_interpretation``, exception messages) stay
  passthrough — those come from the model and aren't UI chrome.
- Closed taxonomies the backend understands in English (``mood``,
  ``section``) keep English *values* and translate only the *display*
  via ``format_func`` lambdas.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

LANGUAGES: dict[str, str] = {"en": "English", "ru": "Русский"}
DEFAULT_LANGUAGE = "en"


def current_language() -> str:
    """Return the active language code, defaulting to English."""
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def t(key: str, **kwargs: Any) -> str:
    """Look up a translation for ``key`` and format with ``kwargs``.

    Resolution order:
        1. Current language's table.
        2. English fallback.
        3. The key itself (so a missing entry is loud, not silent).

    ``str.format`` is applied only when kwargs are supplied; missing
    placeholders in the template degrade to the unformatted template
    rather than raising.
    """
    lang = current_language()
    table = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANGUAGE])
    template = table.get(key)
    if template is None:
        template = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # App-level chrome
        "app.title": "HumMuse",
        "app.caption": "AI co-creative songwriting assistant",
        "language.label": "🌐 Language / Язык",
        # Sidebar
        "sidebar.subtitle": "Session-first songwriting workspace",
        "sidebar.new_session": "New Session",
        "sidebar.refresh": "Refresh",
        "sidebar.load_existing": "Load Existing Session",
        "sidebar.option_none": "None",
        "sidebar.load_session": "Load Session",
        "sidebar.created_session": "Created session {sid}",
        "sidebar.could_not_create": "Could not create session: {err}",
        "sidebar.refreshed": "Session list refreshed",
        "sidebar.could_not_load_list": "Could not load sessions: {err}",
        "sidebar.choose_session_first": "Choose a session from the list first.",
        "sidebar.loaded_session": "Loaded session {sid}",
        "sidebar.could_not_load": "Could not load session: {err}",
        "sidebar.active_session": "Active Session",
        "sidebar.create_or_load": "Create or load a session to begin.",
        # History panel
        "history.title": "Version history",
        "history.no_snapshots": "No snapshots yet.",
        "history.single_snapshot": "1 snapshot — {label}",
        "history.browse": "Browse versions",
        "history.saved_at": "Saved {time}",
        "history.this_is_live": "This is the current live state.",
        "history.viewing": "Viewing v{n} of {total} — restore to make it live.",
        "history.restore": "Restore this version",
        "history.restore_help": "Roll the live song state back to this snapshot.",
        "history.restored": "Restored v{n}.",
        # Snapshot labels (rendered through t() at display time)
        "snapshot.initial": "Initial (empty session)",
        "snapshot.loaded": "Loaded session",
        "snapshot.extracted_melody": "Extracted melody ({n} notes)",
        "snapshot.generated_chords": "Generated chords ({n} progressions, mood={mood})",
        "snapshot.manual_progression": "Manual progression: {chords}",
        "snapshot.accepted_continuation": "Accepted melody continuation #{n}",
        "snapshot.selected_progression": "Selected progression v{n}: {title}",
        "snapshot.refined": "Refined {target}: '{instruction}'",
        "snapshot.restored": "Restored v{n}: {label}",
        "snapshot.version_label": "v{n}: {label}",
        # Session overview
        "overview.use_sidebar": "Use the sidebar to create a session or load an existing one.",
        "overview.session": "Session",
        "overview.melody_notes": "Melody Notes",
        "overview.lyric_suggestions": "Lyric Suggestions",
        "overview.chat_turns": "Chat Turns",
        "overview.current_session_id": "Current session id: `{sid}`",
        "overview.latest_refinement": "Latest refinement for `{target}`: {text}",
        "overview.interpretation": "Interpretation: {text}",
        # Story Bible panel
        "story.expander_title": "📓 Story Bible — creative context",
        "story.caption": (
            "Central creative-intent fields. The graph below shows which "
            "fields drive which subsystems — every generation can be traced "
            "back to its inputs here. Snapshots in the sidebar capture this "
            "alongside the artifact."
        ),
        "story.mood_label": "Mood",
        "story.mood_help": (
            "Drives DQN chord candidate scoring, melody continuation "
            "temperature, and the emotion vector passed to lyric prompts."
        ),
        "story.detected_header": "**Auto-detected from audio**",
        "story.detected_key": "Key",
        "story.detected_tempo": "Tempo",
        "story.graph_expander": "Dependency graph",
        "story.graph_caption": (
            "Yellow = user-controlled intent · Blue = detected from audio · "
            "Green = generative subsystems. Edge labels name what flows "
            "along each edge, and every edge corresponds to a real code "
            "path in the repo."
        ),
        # Main page refresh
        "main.refresh_active": "Refresh Active Session",
        "main.refreshed": "Session refreshed",
        "main.could_not_refresh": "Could not refresh session: {err}",
        # Tab labels
        "tab.melody": "Melody",
        "tab.chords": "Chords",
        "tab.lyrics": "Lyrics",
        "tab.explanations": "Explanations",
        # Melody tab
        "melody.subheader": "Melody",
        "melody.context_placeholder": "Session context will appear here after you create or load one.",
        "melody.upload_label": "Upload or record a melody sketch",
        "melody.record": "Record humming",
        "melody.upload": "Or upload an audio file",
        "melody.upload_help": "Accepted formats: WAV, WebM, MP3",
        "melody.source_recording": "Using your recorded humming. Upload a file or clear the recording to switch.",
        "melody.source_upload": "Using uploaded file: **{name}**. Re-record or clear the upload to switch.",
        "melody.mood_caption": "Mood: **{mood}** (single source of truth — edit in the Story Bible panel above).",
        "melody.extract_button": "Extract Melody",
        "melody.extracted_n": "Extracted {n} note events.",
        "melody.could_not_extract": "Could not extract melody: {err}",
        "melody.current_notes": "Current extracted notes",
        "melody.confidence_legend": "Confidence colors: green = strong, amber = usable, red = low confidence.",
        "melody.no_notes_yet": "No melody extracted yet. Record or upload audio, then click `Extract Melody`.",
        "melody.profile_header": "Melody profile",
        "melody.latest_extraction": "Latest extraction",
        "melody.detected_key": "Detected Key",
        "melody.detected_tempo": "Detected Tempo",
        "melody.returned_notes": "Returned Notes",
        "melody.unknown": "Unknown",
        "melody.pipeline_timings": "Pipeline timings",
        "melody.play": "Play Melody",
        "melody.download_midi": "Download MIDI",
        "melody.rendered_via": "Rendered playback using {source}.",
        "melody.could_not_render": "Could not render playback: {err}",
        "melody.playback_source": "Playback source: `{source}`",
        # Note table headers
        "notes.pitch": "Pitch",
        "notes.onset": "Onset (beats)",
        "notes.duration": "Duration (beats)",
        "notes.velocity": "Velocity",
        "notes.confidence": "Confidence",
        # Chords tab
        "chords.subheader": "Chords",
        "chords.context_placeholder": "Chord generation will use the active session once selected.",
        "chords.generate_from_lyrics_header": "**Generate chords from lyrics**",
        "chords.lyrics_label": "Lyrics",
        "chords.lyrics_placeholder": "Type or paste your lyrics here, then generate candidate progressions.",
        "chords.generate_button": "Generate Chords",
        "chords.generated_n": "Generated {n} candidate progressions for a {mood} mood.",
        "chords.could_not_generate": "Could not generate chords: {err}",
        "chords.manual_header": "**Type your own progression**",
        "chords.manual_caption": (
            "Bypass lyric/DQN inference and supply a progression directly. "
            "Separate chords with commas, arrows (->), pipes (|), or whitespace. "
            "Phase 5 melody continuation will constrain candidates to fit it."
        ),
        "chords.progression_label": "Progression",
        "chords.progression_placeholder": "e.g. C, F, G, C",
        "chords.parsed": "Parsed: {chords}",
        "chords.use_progression": "Use this progression",
        "chords.saved_manual": "Saved user-supplied progression ({n} chords).",
        "chords.could_not_save": "Could not save progression: {err}",
        "chords.candidate_header": "**Candidate progressions**",
        "chords.candidate_caption": (
            "Pick one to commit it as the active progression — the others "
            "stay in version history if you want to roll back."
        ),
        "chords.variant_label": "**Variant {n}**",
        "chords.committed_variant": "Committed variant {n}.",
        "chords.could_not_commit": "Could not commit progression: {err}",
        "chords.no_progressions": "No chord progressions yet.",
        # Suggestions tab
        "suggestions.subheader": "Suggestions",
        "suggestions.placeholder": "Suggestion tools become available once a session is active.",
        "suggestions.melody_header": "**Melody continuation**",
        "suggestions.primer_label": "I hummed a…",
        "suggestions.primer_help": (
            "Optional song-form label for the primer you hummed. "
            "Leave as 'Any' to continue stylistically; pick a section "
            "together with a target to apply BiMMuDa transition priors."
        ),
        "suggestions.target_label": "Generate a…",
        "suggestions.target_help": (
            "Optional song-form target for the continuation. Together "
            "with the primer label, switches on section-conditional "
            "constraints (e.g. verse → chorus lifts register)."
        ),
        "suggestions.generate_melody": "Generate Melody Continuations",
        "suggestions.generated_melody_n": "Generated {n} melody options.",
        "suggestions.could_not_generate_melody": "Could not generate melody continuations: {err}",
        "suggestions.coherence": "Coherence",
        "suggestions.avg_log_p": "Avg log P",
        "suggestions.engine": "Engine",
        "suggestions.preview_unavailable": "Preview unavailable for this continuation.",
        "suggestions.use_continuation": "Use this continuation",
        "suggestions.melody_extended": "Melody extended.",
        "suggestions.could_not_apply": "Could not apply melody continuation: {err}",
        "suggestions.no_continuations": "No melody continuations yet.",
        "suggestions.lyric_header": "**Lyric suggestions**",
        "suggestions.generate_lyrics": "Generate Lyric Suggestions",
        "suggestions.generated_lyrics_n": "Generated {n} lyric options.",
        "suggestions.could_not_generate_lyrics": "Could not generate lyric suggestions: {err}",
        "suggestions.lyric_caption": "Mode: {mode} | Syllables: {syllables}",
        "suggestions.no_lyrics": "No lyric suggestions yet.",
        # Explanations tab
        "explanations.subheader": "Explanations",
        "explanations.placeholder": "Explanation tools become available once a session is active.",
        "explanations.xai_header": "**Structured XAI data**",
        "explanations.no_report": "No explanation report yet. Run a melody, chord, or suggestion action first.",
        "explanations.chat_header": "**Chat panel**",
        "explanations.you": "You",
        "explanations.bot": "HumMuse",
        "explanations.placeholder_ask": "Ask why the system chose something after generating some material.",
        "explanations.input_label": "Ask about the current explanation report",
        "explanations.input_placeholder": "Why did you choose Dm here?",
        "explanations.send_button": "Send Question",
        "explanations.replied": "Received explanation reply.",
        "explanations.could_not_send": "Could not send explanation question: {err}",
        # Refinement panel
        "refine.header": "**Refine This Section**",
        "refine.placeholder": "Refinement instruction",
        "refine.example_placeholder": "Example: make it jazzier",
        "refine.button.melody": "Refine Melody",
        "refine.button.chords": "Refine Chords",
        "refine.button.suggestions": "Refine Suggestions",
        "refine.applied": "Applied refinement to {target}.",
        "refine.could_not": "Could not refine {target}: {err}",
        "refine.interpretation": "Interpretation: {text}",
        # Refinement targets (used in dynamic messages)
        "target.melody": "melody",
        "target.chords": "chords",
        "target.suggestions": "suggestions",
        "target.session": "session",
        # Mood display (values stay English; only the label changes)
        "mood.uplift": "uplift",
        "mood.melancholy": "melancholy",
        "mood.tense": "tense",
        "mood.calm": "calm",
        "mood.energetic": "energetic",
        "mood.romantic": "romantic",
        "mood.dark": "dark",
        "mood.playful": "playful",
        # Section display
        "section.Any": "Any",
        "section.verse": "verse",
        "section.pre_chorus": "pre-chorus",
        "section.chorus": "chorus",
        "section.bridge": "bridge",
        # Lyrics tab (split out from old Suggestions tab)
        "lyrics.subheader": "Lyrics",
        "lyrics.placeholder": "Lyrics tools become available once a session is active.",
        "lyrics.input_label": "Lyrics",
        "lyrics.input_placeholder": "Type or paste your lyrics here. They feed both chord generation and lyric suggestions.",
        "lyrics.input_help": (
            "Single source of truth for lyrics. Persisted to session state "
            "when you click Generate; also used by the Chords tab."
        ),
        "lyrics.generate_button": "Generate Lyric Suggestions",
        "lyrics.generated_n": "Generated {n} lyric options.",
        "lyrics.could_not_generate": "Could not generate lyric suggestions: {err}",
        "lyrics.no_suggestions": "No lyric suggestions yet.",
        # Chord tab: lyrics now live in their own tab; preview here
        "chords.lyrics_preview_header": "**Lyrics (from Lyrics tab)**",
        "chords.lyrics_empty_hint": "Add lyrics in the **Lyrics** tab to enable chord generation from text.",
        # Refinement: "lyrics" replaces "suggestions" as the target name
        "target.lyrics": "lyrics",
        "refine.button.lyrics": "Refine Lyrics",
        # Snapshot label for lyric updates committed via the Lyrics tab
        "snapshot.generated_lyrics": "Generated lyrics ({n} variants)",
    },
    "ru": {
        # App-level chrome
        "app.title": "HumMuse",
        "app.caption": "ИИ-ассистент для совместного написания песен",
        "language.label": "🌐 Language / Язык",
        # Sidebar
        "sidebar.subtitle": "Рабочее пространство для песен на основе сессий",
        "sidebar.new_session": "Новая сессия",
        "sidebar.refresh": "Обновить",
        "sidebar.load_existing": "Загрузить существующую сессию",
        "sidebar.option_none": "Нет",
        "sidebar.load_session": "Загрузить сессию",
        "sidebar.created_session": "Сессия создана: {sid}",
        "sidebar.could_not_create": "Не удалось создать сессию: {err}",
        "sidebar.refreshed": "Список сессий обновлён",
        "sidebar.could_not_load_list": "Не удалось загрузить список сессий: {err}",
        "sidebar.choose_session_first": "Сначала выберите сессию из списка.",
        "sidebar.loaded_session": "Сессия загружена: {sid}",
        "sidebar.could_not_load": "Не удалось загрузить сессию: {err}",
        "sidebar.active_session": "Активная сессия",
        "sidebar.create_or_load": "Создайте или загрузите сессию, чтобы начать.",
        # History panel
        "history.title": "История версий",
        "history.no_snapshots": "Снимков пока нет.",
        "history.single_snapshot": "1 снимок — {label}",
        "history.browse": "Просмотр версий",
        "history.saved_at": "Сохранено: {time}",
        "history.this_is_live": "Это текущее состояние.",
        "history.viewing": "Просмотр v{n} из {total} — восстановите, чтобы сделать активным.",
        "history.restore": "Восстановить эту версию",
        "history.restore_help": "Откатить состояние песни к этому снимку.",
        "history.restored": "Восстановлено: v{n}.",
        # Snapshot labels
        "snapshot.initial": "Начальное (пустая сессия)",
        "snapshot.loaded": "Сессия загружена",
        "snapshot.extracted_melody": "Извлечена мелодия ({n} нот)",
        "snapshot.generated_chords": "Сгенерированы аккорды ({n} вариантов, настроение={mood})",
        "snapshot.manual_progression": "Ручная последовательность: {chords}",
        "snapshot.accepted_continuation": "Принято продолжение мелодии №{n}",
        "snapshot.selected_progression": "Выбрана прогрессия v{n}: {title}",
        "snapshot.refined": "Доработка ({target}): «{instruction}»",
        "snapshot.restored": "Восстановлено v{n}: {label}",
        "snapshot.version_label": "v{n}: {label}",
        # Session overview
        "overview.use_sidebar": "Используйте боковую панель, чтобы создать или загрузить сессию.",
        "overview.session": "Сессия",
        "overview.melody_notes": "Ноты мелодии",
        "overview.lyric_suggestions": "Варианты текста",
        "overview.chat_turns": "Сообщений в чате",
        "overview.current_session_id": "ID текущей сессии: `{sid}`",
        "overview.latest_refinement": "Последняя доработка для `{target}`: {text}",
        "overview.interpretation": "Интерпретация: {text}",
        # Story Bible panel
        "story.expander_title": "📓 Story Bible — творческий контекст",
        "story.caption": (
            "Центральные поля творческого замысла. Граф ниже показывает, "
            "какие поля влияют на какие подсистемы — каждое порождение "
            "можно проследить до его входов. Снимки в боковой панели "
            "сохраняют это вместе с артефактом."
        ),
        "story.mood_label": "Настроение",
        "story.mood_help": (
            "Влияет на оценку аккордов в DQN, температуру продолжения "
            "мелодии и эмоциональный вектор для генерации текстов."
        ),
        "story.detected_header": "**Определено автоматически из аудио**",
        "story.detected_key": "Тональность",
        "story.detected_tempo": "Темп",
        "story.graph_expander": "Граф зависимостей",
        "story.graph_caption": (
            "Жёлтый = намерения пользователя · Синий = определено из аудио · "
            "Зелёный = генеративные подсистемы. Подписи рёбер показывают, "
            "что передаётся по каждому ребру; всё соответствует реальному "
            "коду в репозитории."
        ),
        # Main page refresh
        "main.refresh_active": "Обновить активную сессию",
        "main.refreshed": "Сессия обновлена",
        "main.could_not_refresh": "Не удалось обновить сессию: {err}",
        # Tab labels
        "tab.melody": "Мелодия",
        "tab.chords": "Аккорды",
        "tab.lyrics": "Тексты",
        "tab.explanations": "Объяснения",
        # Melody tab
        "melody.subheader": "Мелодия",
        "melody.context_placeholder": "Контекст сессии появится здесь после её создания или загрузки.",
        "melody.upload_label": "Загрузите или запишите мелодический набросок",
        "melody.record": "Записать напев",
        "melody.upload": "Или загрузите аудиофайл",
        "melody.upload_help": "Поддерживаемые форматы: WAV, WebM, MP3",
        "melody.source_recording": "Используется запись напева. Загрузите файл или очистите запись, чтобы переключиться.",
        "melody.source_upload": "Используется загруженный файл: **{name}**. Перезапишите напев или очистите загрузку, чтобы переключиться.",
        "melody.mood_caption": "Настроение: **{mood}** (единственный источник истины — редактируется в Story Bible выше).",
        "melody.extract_button": "Извлечь мелодию",
        "melody.extracted_n": "Извлечено нот: {n}.",
        "melody.could_not_extract": "Не удалось извлечь мелодию: {err}",
        "melody.current_notes": "Текущие извлечённые ноты",
        "melody.confidence_legend": "Цвета уверенности: зелёный = высокая, жёлтый = приемлемая, красный = низкая.",
        "melody.no_notes_yet": "Мелодия ещё не извлечена. Запишите или загрузите аудио и нажмите «Извлечь мелодию».",
        "melody.profile_header": "Профиль мелодии",
        "melody.latest_extraction": "Последнее извлечение",
        "melody.detected_key": "Определённая тональность",
        "melody.detected_tempo": "Определённый темп",
        "melody.returned_notes": "Возвращено нот",
        "melody.unknown": "Неизвестно",
        "melody.pipeline_timings": "Время выполнения пайплайна",
        "melody.play": "Воспроизвести мелодию",
        "melody.download_midi": "Скачать MIDI",
        "melody.rendered_via": "Воспроизведение через: {source}.",
        "melody.could_not_render": "Не удалось воспроизвести: {err}",
        "melody.playback_source": "Источник воспроизведения: `{source}`",
        # Note table headers
        "notes.pitch": "Высота",
        "notes.onset": "Начало (доли)",
        "notes.duration": "Длительность (доли)",
        "notes.velocity": "Громкость",
        "notes.confidence": "Уверенность",
        # Chords tab
        "chords.subheader": "Аккорды",
        "chords.context_placeholder": "Генерация аккордов станет доступна после активации сессии.",
        "chords.generate_from_lyrics_header": "**Сгенерировать аккорды из текста**",
        "chords.lyrics_label": "Текст песни",
        "chords.lyrics_placeholder": "Введите или вставьте текст здесь, затем сгенерируйте варианты прогрессий.",
        "chords.generate_button": "Сгенерировать аккорды",
        "chords.generated_n": "Сгенерировано вариантов прогрессий: {n} для настроения «{mood}».",
        "chords.could_not_generate": "Не удалось сгенерировать аккорды: {err}",
        "chords.manual_header": "**Введите свою последовательность**",
        "chords.manual_caption": (
            "Обойти вывод из текста/DQN и задать прогрессию напрямую. "
            "Разделяйте аккорды запятыми, стрелками (->), вертикальной чертой (|) "
            "или пробелами. Продолжение мелодии (Phase 5) будет учитывать её."
        ),
        "chords.progression_label": "Последовательность",
        "chords.progression_placeholder": "например: C, F, G, C",
        "chords.parsed": "Распознано: {chords}",
        "chords.use_progression": "Использовать эту последовательность",
        "chords.saved_manual": "Сохранена ручная последовательность ({n} аккордов).",
        "chords.could_not_save": "Не удалось сохранить последовательность: {err}",
        "chords.candidate_header": "**Варианты прогрессий**",
        "chords.candidate_caption": (
            "Выберите один, чтобы зафиксировать его как активную прогрессию — "
            "остальные сохранятся в истории версий для отката."
        ),
        "chords.variant_label": "**Вариант {n}**",
        "chords.committed_variant": "Зафиксирован вариант {n}.",
        "chords.could_not_commit": "Не удалось зафиксировать прогрессию: {err}",
        "chords.no_progressions": "Прогрессий пока нет.",
        # Suggestions tab
        "suggestions.subheader": "Подсказки",
        "suggestions.placeholder": "Инструменты подсказок появятся после активации сессии.",
        "suggestions.melody_header": "**Продолжение мелодии**",
        "suggestions.primer_label": "Я напел…",
        "suggestions.primer_help": (
            "Необязательная метка формы песни для напетого фрагмента. "
            "Оставьте «Любая», чтобы продолжить стилистически; выберите "
            "секцию вместе с целевой — применятся переходные приоры BiMMuDa."
        ),
        "suggestions.target_label": "Сгенерировать…",
        "suggestions.target_help": (
            "Необязательная метка формы песни для продолжения. Вместе с "
            "меткой напева включает условные ограничения по секциям "
            "(например, куплет → припев поднимает регистр)."
        ),
        "suggestions.generate_melody": "Сгенерировать продолжения мелодии",
        "suggestions.generated_melody_n": "Сгенерировано вариантов мелодии: {n}.",
        "suggestions.could_not_generate_melody": "Не удалось сгенерировать продолжения: {err}",
        "suggestions.coherence": "Связность",
        "suggestions.avg_log_p": "Ср. log P",
        "suggestions.engine": "Движок",
        "suggestions.preview_unavailable": "Предпрослушивание недоступно для этого варианта.",
        "suggestions.use_continuation": "Использовать это продолжение",
        "suggestions.melody_extended": "Мелодия продолжена.",
        "suggestions.could_not_apply": "Не удалось применить продолжение: {err}",
        "suggestions.no_continuations": "Продолжений пока нет.",
        "suggestions.lyric_header": "**Варианты текста**",
        "suggestions.generate_lyrics": "Сгенерировать варианты текста",
        "suggestions.generated_lyrics_n": "Сгенерировано вариантов текста: {n}.",
        "suggestions.could_not_generate_lyrics": "Не удалось сгенерировать варианты текста: {err}",
        "suggestions.lyric_caption": "Режим: {mode} | Слогов: {syllables}",
        "suggestions.no_lyrics": "Вариантов текста пока нет.",
        # Explanations tab
        "explanations.subheader": "Объяснения",
        "explanations.placeholder": "Инструменты объяснений появятся после активации сессии.",
        "explanations.xai_header": "**Структурированные XAI-данные**",
        "explanations.no_report": "Отчёт ещё не сформирован. Сначала выполните действие с мелодией, аккордами или подсказками.",
        "explanations.chat_header": "**Чат**",
        "explanations.you": "Вы",
        "explanations.bot": "HumMuse",
        "explanations.placeholder_ask": "Спросите, почему система выбрала тот или иной вариант, после генерации.",
        "explanations.input_label": "Задайте вопрос об отчёте объяснений",
        "explanations.input_placeholder": "Почему здесь выбран Dm?",
        "explanations.send_button": "Отправить вопрос",
        "explanations.replied": "Получен ответ-объяснение.",
        "explanations.could_not_send": "Не удалось отправить вопрос: {err}",
        # Refinement panel
        "refine.header": "**Доработать этот раздел**",
        "refine.placeholder": "Инструкция для доработки",
        "refine.example_placeholder": "Например: сделать более джазово",
        "refine.button.melody": "Доработать мелодию",
        "refine.button.chords": "Доработать аккорды",
        "refine.button.suggestions": "Доработать подсказки",
        "refine.applied": "Доработка применена: {target}.",
        "refine.could_not": "Не удалось доработать ({target}): {err}",
        "refine.interpretation": "Интерпретация: {text}",
        # Refinement targets
        "target.melody": "мелодия",
        "target.chords": "аккорды",
        "target.suggestions": "подсказки",
        "target.session": "сессия",
        # Mood display
        "mood.uplift": "вдохновение",
        "mood.melancholy": "меланхолия",
        "mood.tense": "напряжение",
        "mood.calm": "спокойствие",
        "mood.energetic": "энергично",
        "mood.romantic": "романтично",
        "mood.dark": "мрачно",
        "mood.playful": "игриво",
        # Section display
        "section.Any": "Любая",
        "section.verse": "куплет",
        "section.pre_chorus": "предприпев",
        "section.chorus": "припев",
        "section.bridge": "бридж",
        # Lyrics tab (split out from old Suggestions tab)
        "lyrics.subheader": "Тексты",
        "lyrics.placeholder": "Инструменты для текстов появятся после активации сессии.",
        "lyrics.input_label": "Текст песни",
        "lyrics.input_placeholder": "Введите или вставьте текст. Он используется и для генерации аккордов, и для подсказок текстов.",
        "lyrics.input_help": (
            "Единственный источник истины для текста. Сохраняется в "
            "сессию при нажатии «Сгенерировать»; также используется во "
            "вкладке «Аккорды»."
        ),
        "lyrics.generate_button": "Сгенерировать варианты текста",
        "lyrics.generated_n": "Сгенерировано вариантов текста: {n}.",
        "lyrics.could_not_generate": "Не удалось сгенерировать варианты текста: {err}",
        "lyrics.no_suggestions": "Вариантов текста пока нет.",
        # Chord tab: lyrics now live in their own tab; preview here
        "chords.lyrics_preview_header": "**Текст (из вкладки «Тексты»)**",
        "chords.lyrics_empty_hint": "Добавьте текст во вкладке **«Тексты»**, чтобы включить генерацию аккордов из текста.",
        # Refinement
        "target.lyrics": "тексты",
        "refine.button.lyrics": "Доработать тексты",
        # Snapshot label
        "snapshot.generated_lyrics": "Сгенерированы тексты ({n} вариантов)",
    },
}
