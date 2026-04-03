"""Structured JSON logging configuration for API request lifecycle events."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import Request

try:  # pragma: no cover - exercised when structlog is installed in the environment.
    import structlog
except ModuleNotFoundError:  # pragma: no cover - fallback covered in tests.
    structlog = None


_CONFIGURED = False


class _FallbackLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("info", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("error", event, **fields)

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "logger": self._logger.name,
            "event": event,
            **fields,
        }
        getattr(self._logger, level)(json.dumps(payload, ensure_ascii=False))


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[handler], force=True)

    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    _CONFIGURED = True


def get_logger(name: str) -> Any:
    configure_logging()
    if structlog is not None:
        return structlog.get_logger(name)
    return _FallbackLogger(name)


def initialize_request_context(
    request: Request,
    *,
    endpoint: str,
    pipeline_stage: str,
    session_id: str | None = None,
    use_case: str | None = None,
) -> None:
    request.state.log_context = {
        "session_id": session_id,
        "endpoint": endpoint,
        "pipeline_stage": pipeline_stage,
        "use_case": use_case,
    }


def bind_request_context(request: Request, **fields: Any) -> None:
    current = getattr(request.state, "log_context", {})
    current.update(fields)
    request.state.log_context = current


def get_request_context(request: Request) -> dict[str, Any]:
    return dict(getattr(request.state, "log_context", {}))
