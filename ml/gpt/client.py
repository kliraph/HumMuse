"""Provider-neutral OpenAI-compatible chat client for GPT calls."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a project dependency.
    load_dotenv = None


ProviderName = Literal["yandex", "openai"]
ChatMessage = Mapping[str, Any]

DEFAULT_PROVIDER: ProviderName = "yandex"
DEFAULT_YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_YANDEX_MODEL = "gpt://b1gpgmngop2oujkaav6l/yandexgpt-lite/latest@tamr8beao63vjn3o8864v"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5
DEFAULT_DOTENV_PATH = Path(__file__).resolve().with_name(".env")


class LLMConfigError(ValueError):
    """Raised when LLM environment configuration is invalid."""


class GPTClientError(RuntimeError):
    """Raised when the OpenAI-compatible SDK cannot be used."""


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for an OpenAI-compatible LLM provider."""

    provider: ProviderName
    model: str
    api_key: str
    base_url: str | None = None
    project: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "LLMConfig":
        """Load provider configuration from `.env` and process environment."""
        if load_dotenv is not None:
            load_dotenv(dotenv_path=dotenv_path or DEFAULT_DOTENV_PATH, override=False)

        provider = _read_provider()
        timeout_seconds = _read_float_env("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, minimum=0.0)
        max_retries = _read_int_env("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0)
        backoff_seconds = _read_float_env("LLM_BACKOFF_SECONDS", DEFAULT_BACKOFF_SECONDS, minimum=0.0)

        if provider == "yandex":
            return cls(
                provider=provider,
                model=_first_non_empty("LLM_MODEL") or DEFAULT_YANDEX_MODEL,
                api_key=_require_first_non_empty("LLM_API_KEY", "YANDEX_API_KEY", "YANDEX_CLOUD_API_KEY"),
                base_url=_first_non_empty("LLM_BASE_URL") or DEFAULT_YANDEX_BASE_URL,
                project=_first_non_empty("LLM_PROJECT", "YANDEX_FOLDER_ID", "YANDEX_CLOUD_FOLDER"),
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )

        model = _first_non_empty("LLM_MODEL", "OPENAI_MODEL")
        if model is None:
            raise LLMConfigError("OpenAI provider requires LLM_MODEL or OPENAI_MODEL")
        return cls(
            provider=provider,
            model=model,
            api_key=_require_first_non_empty("LLM_API_KEY", "OPENAI_API_KEY"),
            base_url=_first_non_empty("LLM_BASE_URL", "OPENAI_BASE_URL"),
            project=_first_non_empty("LLM_PROJECT", "OPENAI_PROJECT"),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by `GPTClient.complete`."""

    content: str
    model: str
    provider: ProviderName
    usage: dict[str, Any]
    response_id: str | None = None
    metadata: dict[str, Any] | None = None


class GPTClient:
    """Thin retrying wrapper around an OpenAI-compatible chat client."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        raw_client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._client = raw_client if raw_client is not None else self._build_client(client_factory)

    @classmethod
    def from_env(
        cls,
        dotenv_path: str | Path | None = None,
        *,
        client_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "GPTClient":
        """Create a GPT client from `.env` and process environment settings."""
        return cls(LLMConfig.from_env(dotenv_path), client_factory=client_factory, sleep=sleep)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a chat completion request and return normalized text output."""
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            **kwargs,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = dict(response_format)

        response = self._request_with_retries(lambda: self._client.chat.completions.create(**params))
        return LLMResponse(
            content=_extract_content(response),
            model=str(getattr(response, "model", None) or self.config.model),
            provider=self.config.provider,
            usage=_extract_usage(response),
            response_id=getattr(response, "id", None),
            metadata=_extract_metadata(response),
        )

    def _build_client(self, client_factory: Callable[..., Any] | None) -> Any:
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on local environment.
                raise GPTClientError("The OpenAI SDK is required. Install project dependency `openai>=2.36,<3`.") from exc
            client_factory = OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
            "max_retries": 0,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.project:
            kwargs["project"] = self.config.project
        return client_factory(**kwargs)

    def _request_with_retries(self, request: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return request()
            except Exception as exc:  # pragma: no cover - branch coverage comes from tests.
                last_error = exc
                if attempt >= self.config.max_retries or not _is_transient_error(exc):
                    raise
                self._sleep(self.config.backoff_seconds * (2**attempt))
        raise GPTClientError("LLM request failed without returning a response") from last_error


def _read_provider() -> ProviderName:
    raw_provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if raw_provider not in ("yandex", "openai"):
        raise LLMConfigError("LLM_PROVIDER must be either 'yandex' or 'openai'")
    return raw_provider


def _first_non_empty(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        stripped = value.strip()
        if not stripped:
            raise LLMConfigError(f"{name} must not be empty when set")
        return stripped
    return None


def _require_first_non_empty(*names: str) -> str:
    value = _first_non_empty(*names)
    if value is None:
        joined = " or ".join(names)
        raise LLMConfigError(f"Missing required API key: set {joined}")
    return value


def _read_float_env(name: str, default: float, *, minimum: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LLMConfigError(f"{name} must be a number") from exc
    if parsed < minimum:
        raise LLMConfigError(f"{name} must be >= {minimum}")
    return parsed


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LLMConfigError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise LLMConfigError(f"{name} must be >= {minimum}")
    return parsed


def _extract_content(response: Any) -> str:
    choices = _get_value(response, "choices") or []
    if not choices:
        return ""
    first_choice = choices[0]
    message = _get_value(first_choice, "message") or {}
    content = _get_value(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _get_value(item, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = _get_value(response, "usage")
    plain = _to_plain(usage)
    return plain if isinstance(plain, dict) else {}


def _extract_metadata(response: Any) -> dict[str, Any]:
    metadata = {
        "object": getattr(response, "object", None),
        "created": getattr(response, "created", None),
        "finish_reason": None,
    }
    choices = _get_value(response, "choices") or []
    if choices:
        metadata["finish_reason"] = _get_value(choices[0], "finish_reason")
    return {key: value for key, value in metadata.items() if value is not None}


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500

    name = exc.__class__.__name__.lower()
    if "timeout" in name or "connection" in name:
        return True

    message = str(exc).lower()
    return "timed out" in message or "connection" in message
