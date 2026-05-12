"""Tests for the OpenAI-compatible GPT provider client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

import ml.gpt.client as client_module
from ml.gpt.client import (
    DEFAULT_DOTENV_PATH,
    DEFAULT_YANDEX_BASE_URL,
    DEFAULT_YANDEX_MODEL,
    GPTClient,
    LLMConfig,
    LLMConfigError,
)


ENV_KEYS = [
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROJECT",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_BACKOFF_SECONDS",
    "YANDEX_API_KEY",
    "YANDEX_CLOUD_API_KEY",
    "YANDEX_FOLDER_ID",
    "YANDEX_CLOUD_FOLDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_PROJECT",
]


class TransientError(RuntimeError):
    status_code = 500


class RateLimitError(RuntimeError):
    status_code = 429


class ClientError(RuntimeError):
    status_code = 400


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def model_dump(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class FakeCompletion:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChat:
    def __init__(self, completion: FakeCompletion) -> None:
        self.completions = completion


class FakeOpenAIClient:
    def __init__(self, responses: list[object]) -> None:
        self.completion = FakeCompletion(responses)
        self.chat = FakeChat(self.completion)


def fake_response(content: str = "hello", *, model: str = "provider-model") -> object:
    return type(
        "FakeResponse",
        (),
        {
            "id": "resp_123",
            "model": model,
            "object": "chat.completion",
            "created": 123,
            "usage": FakeUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )()


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(client_module, "load_dotenv", lambda *_, **__: None)


def test_yandex_defaults_use_model_base_url_and_generic_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    config = LLMConfig.from_env()

    assert config.provider == "yandex"
    assert config.model == DEFAULT_YANDEX_MODEL
    assert config.base_url == DEFAULT_YANDEX_BASE_URL
    assert config.api_key == "test-key"


def test_env_file_loading_keeps_process_env_precedence(monkeypatch) -> None:
    dotenv_path = Path(".env.test")

    def fake_load_dotenv(dotenv_path=None, override=False):
        assert dotenv_path == Path(".env.test")
        assert override is False
        os.environ.setdefault("LLM_PROVIDER", "yandex")
        os.environ.setdefault("YANDEX_API_KEY", "file-key")
        os.environ.setdefault("LLM_MODEL", "file-model")
        os.environ.setdefault("LLM_PROJECT", "file-folder")

    monkeypatch.setattr(client_module, "load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("LLM_MODEL", "process-model")

    config = LLMConfig.from_env(dotenv_path)

    assert config.api_key == "file-key"
    assert config.model == "process-model"
    assert config.project == "file-folder"


def test_default_env_file_is_module_local(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dotenv(dotenv_path=None, override=False):
        captured["dotenv_path"] = dotenv_path
        captured["override"] = override

    monkeypatch.setattr(client_module, "load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    config = LLMConfig.from_env()

    assert config.provider == "yandex"
    assert captured == {"dotenv_path": DEFAULT_DOTENV_PATH, "override": False}


def test_openai_provider_can_be_selected_by_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    config = LLMConfig.from_env()

    assert config.provider == "openai"
    assert config.api_key == "openai-key"
    assert config.model == "gpt-test"
    assert config.base_url == "https://example.test/v1"


def test_invalid_provider_raises_config_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(LLMConfigError, match="LLM_PROVIDER"):
        LLMConfig.from_env()


def test_missing_yandex_key_raises_config_error() -> None:
    with pytest.raises(LLMConfigError, match="Missing required API key"):
        LLMConfig.from_env()


def test_openai_provider_requires_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with pytest.raises(LLMConfigError, match="OPENAI_MODEL"):
        LLMConfig.from_env()


def test_invalid_numeric_setting_raises_config_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "slow")

    with pytest.raises(LLMConfigError, match="LLM_TIMEOUT_SECONDS"):
        LLMConfig.from_env()


def test_complete_returns_normalized_response() -> None:
    raw_client = FakeOpenAIClient([fake_response("grounded answer", model="actual-model")])
    client = GPTClient(
        LLMConfig(provider="yandex", model="configured-model", api_key="key", base_url=DEFAULT_YANDEX_BASE_URL),
        raw_client=raw_client,
    )

    result = client.complete(
        [{"role": "user", "content": "Why Dm here?"}],
        temperature=0.2,
        max_tokens=64,
    )

    assert result.content == "grounded answer"
    assert result.model == "actual-model"
    assert result.provider == "yandex"
    assert result.response_id == "resp_123"
    assert result.usage["total_tokens"] == 7
    assert result.metadata == {"object": "chat.completion", "created": 123, "finish_reason": "stop"}
    assert raw_client.completion.calls[0]["model"] == "configured-model"
    assert raw_client.completion.calls[0]["temperature"] == 0.2
    assert raw_client.completion.calls[0]["max_tokens"] == 64


def test_transient_errors_are_retried() -> None:
    raw_client = FakeOpenAIClient([TransientError("temporary"), RateLimitError("rate limited"), fake_response("ok")])
    sleeps: list[float] = []
    client = GPTClient(
        LLMConfig(
            provider="yandex",
            model="configured-model",
            api_key="key",
            max_retries=2,
            backoff_seconds=0.25,
        ),
        raw_client=raw_client,
        sleep=sleeps.append,
    )

    result = client.complete([{"role": "user", "content": "retry"}])

    assert result.content == "ok"
    assert len(raw_client.completion.calls) == 3
    assert sleeps == [0.25, 0.5]


def test_non_transient_errors_are_not_retried() -> None:
    raw_client = FakeOpenAIClient([ClientError("bad request"), fake_response("should not happen")])
    client = GPTClient(
        LLMConfig(provider="yandex", model="configured-model", api_key="key", max_retries=2),
        raw_client=raw_client,
    )

    with pytest.raises(ClientError):
        client.complete([{"role": "user", "content": "fail"}])

    assert len(raw_client.completion.calls) == 1


def test_client_factory_receives_provider_config() -> None:
    captured_kwargs: dict[str, object] = {}

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeOpenAIClient([fake_response("ok")])

    client = GPTClient(
        LLMConfig(
            provider="yandex",
            model="configured-model",
            api_key="key",
            base_url=DEFAULT_YANDEX_BASE_URL,
            project="folder-id",
            timeout_seconds=12.0,
        ),
        client_factory=factory,
    )

    assert client.complete([{"role": "user", "content": "hello"}]).content == "ok"
    assert captured_kwargs == {
        "api_key": "key",
        "base_url": DEFAULT_YANDEX_BASE_URL,
        "project": "folder-id",
        "timeout": 12.0,
        "max_retries": 0,
    }
