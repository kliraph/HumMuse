"""Session history + chat-message utility helpers.

These functions are NOT mock implementations, they are the production
helpers every API endpoint uses to append :class:`Action` rows to
``state.history`` and :class:`ChatMessage` rows to ``state.chat_history``.
They previously lived in ``backend/mock_pipeline.py`` for historical
reasons, but that module is now strictly GPT-down fallbacks, so the live
helpers moved here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from shared.schemas import Action, ChatMessage, SessionState


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def add_history(
    state: SessionState,
    kind: str,
    payload: dict[str, object],
    *,
    source: str = "api",
) -> SessionState:
    """Append an :class:`Action` row to ``state.history`` and return the state."""
    state.history.append(Action(kind=kind, payload=payload, source=source, timestamp=now_iso()))
    return state


def append_chat_message(state: SessionState, role: str, content: str) -> ChatMessage:
    """Append a :class:`ChatMessage` to ``state.chat_history`` and return it."""
    message = ChatMessage(role=role, content=content, timestamp=now_iso())
    state.chat_history.append(message)
    return message
