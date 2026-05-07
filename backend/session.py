"""Session persistence helpers for the shared SessionState contract."""

from __future__ import annotations

from uuid import UUID, uuid4

from backend.database import Database
from shared.schemas import SessionState, SessionSummary


class SessionNotFoundError(KeyError):
    """Raised when a session id is not present in storage."""


class SessionManager:
    def __init__(self, database: Database | None = None, *, pipeline_version: str = "v3.3") -> None:
        self.database = database or Database()
        self.pipeline_version = pipeline_version

    def create_session(self) -> SessionState:
        state = SessionState(session_id=uuid4())
        self.database.upsert_session(
            str(state.session_id),
            state.model_dump_json(),
            self.pipeline_version,
        )
        return state

    def get_session(self, session_id: str | UUID) -> SessionState:
        row = self.database.get_session_row(str(session_id))
        if row is None:
            raise SessionNotFoundError(f"Session {session_id} was not found")
        return SessionState.model_validate_json(row["state_json"])

    def update_session(self, session_id: str | UUID, state: SessionState) -> None:
        session_id_str = str(session_id)
        if str(state.session_id) != session_id_str:
            raise ValueError("SessionState.session_id must match the session being updated")

        row = self.database.get_session_row(session_id_str)
        if row is None:
            raise SessionNotFoundError(f"Session {session_id} was not found")

        self.database.upsert_session(
            session_id_str,
            state.model_dump_json(),
            row["pipeline_version"],
            created_at=row["created_at"],
        )

    def list_sessions(self) -> list[SessionSummary]:
        return [
            SessionSummary(
                session_id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                pipeline_version=row["pipeline_version"],
            )
            for row in self.database.list_session_rows()
        ]
