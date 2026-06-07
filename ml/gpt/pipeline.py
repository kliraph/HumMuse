"""Unified GPT pipeline routing, caching, eval mode, and call logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from ml.gpt.client import GPTClient, LLMResponse
from ml.gpt.context_builder import SessionContext, build_session_context
from ml.gpt.prompts.explain import build_explanation_prompt
from ml.gpt.prompts.lyrics import build_lyric_prompt
from ml.gpt.prompts.mood import build_mood_prompt
from ml.gpt.prompts.refine import build_refinement_prompt, parse_refinement_plan
from shared.schemas import SessionState

GPTUseCase = Literal["lyric", "refine", "explain", "mood"]
LOGGER = logging.getLogger(__name__)
DEFAULT_EVAL_SEED = 0


@dataclass(frozen=True)
class GPTPipelineConfig:
    """Runtime controls for GPT routing."""

    eval_mode: bool = False
    eval_seed: int = DEFAULT_EVAL_SEED
    cache_enabled: bool = True

    @classmethod
    def from_env(cls) -> "GPTPipelineConfig":
        return cls(
            eval_mode=_env_bool("LLM_EVAL_MODE", default=False),
            eval_seed=_env_int("LLM_EVAL_SEED", default=DEFAULT_EVAL_SEED),
            cache_enabled=_env_bool("LLM_CACHE_ENABLED", default=True),
        )


@dataclass(frozen=True)
class GPTPipelineResult:
    """Normalized output from one routed GPT request."""

    use_case: GPTUseCase
    content: str
    parsed: Any
    cache_hit: bool
    provider: str
    model_version: str
    latency_ms: float
    prompt_token_count: int | None
    completion_token_count: int | None
    eval_mode: bool


class ResponseCache:
    """Thin adapter around the existing SQLite api_cache helpers."""

    def __init__(self, database: Any) -> None:
        self.database = database

    def get(self, prompt_hash: str, model_version: str, use_case: str) -> str | None:
        return self.database.get_cached_response(prompt_hash, model_version, use_case)

    def put(self, prompt_hash: str, response_json: str, model_version: str, use_case: str) -> None:
        self.database.cache_response(prompt_hash, response_json, model_version, use_case)


class GPTPipeline:
    """Route lyric, refinement, and explanation GPT requests through one path."""

    def __init__(
        self,
        client: GPTClient,
        *,
        cache: Any | None = None,
        config: GPTPipelineConfig | None = None,
        logger: Any | None = LOGGER,
        clock: Any = time.perf_counter,
    ) -> None:
        self.client = client
        self.cache = cache
        self.config = config or GPTPipelineConfig.from_env()
        self.logger = logger
        self.clock = clock

    @classmethod
    def from_env(cls, *, cache: Any | None = None, logger: Any | None = LOGGER) -> "GPTPipeline":
        return cls(GPTClient.from_env(), cache=cache, config=GPTPipelineConfig.from_env(), logger=logger)

    def route_request(
        self,
        use_case: GPTUseCase,
        session_state: SessionState,
        user_input: str,
        *,
        mode: str = "Poetic",
        num_suggestions: int = 3,
        syllable_targets_per_line: list[int] | None = None,
        target_hint: str | None = None,
        language: str | None = None,
    ) -> GPTPipelineResult:
        prompt = self._build_prompt(
            use_case,
            session_state,
            user_input,
            mode=mode,
            num_suggestions=num_suggestions,
            syllable_targets_per_line=syllable_targets_per_line,
            target_hint=target_hint,
            language=language,
        )
        request_payload = self._request_payload(use_case, prompt, user_input)
        prompt_hash = _hash_payload(request_payload)
        model_version = self.client.config.model
        cache_allowed = self._cache_allowed(use_case)
        started_at = self.clock()

        if cache_allowed and self.cache is not None:
            cached = self.cache.get(prompt_hash, model_version, use_case)
            if cached is not None:
                latency_ms = _elapsed_ms(started_at, self.clock())
                result = self._result_from_cached(use_case, cached, latency_ms=latency_ms)
                self._log_call(result)
                return result

        call_kwargs = self._call_kwargs(prompt)
        if self.config.eval_mode:
            call_kwargs["temperature"] = 0
            call_kwargs["seed"] = self.config.eval_seed

        response = self.client.complete(prompt.messages, **call_kwargs)
        latency_ms = _elapsed_ms(started_at, self.clock())
        parsed = self._parse_response(use_case, response.content)
        result = GPTPipelineResult(
            use_case=use_case,
            content=response.content,
            parsed=parsed,
            cache_hit=False,
            provider=response.provider,
            model_version=response.model,
            latency_ms=latency_ms,
            prompt_token_count=_token_count(response.usage, "prompt_tokens"),
            completion_token_count=_token_count(response.usage, "completion_tokens"),
            eval_mode=self.config.eval_mode,
        )

        if cache_allowed and self.cache is not None:
            self.cache.put(prompt_hash, _serialize_cached_result(result), model_version, use_case)

        self._log_call(result)
        return result

    def _build_prompt(
        self,
        use_case: GPTUseCase,
        session_state: SessionState,
        user_input: str,
        *,
        mode: str,
        num_suggestions: int,
        syllable_targets_per_line: list[int] | None,
        target_hint: str | None,
        language: str | None = None,
    ) -> Any:
        if use_case == "lyric":
            context = build_session_context(
                "lyric",
                session_state,
                syllable_targets_per_line=syllable_targets_per_line,
            )
            return build_lyric_prompt(
                context,
                mode=mode,
                num_suggestions=num_suggestions,
                extra_instruction=user_input,
            )
        if use_case == "refine":
            context = build_session_context(
                "refine",
                session_state,
                syllable_targets_per_line=syllable_targets_per_line,
            )
            return build_refinement_prompt(context, instruction=user_input, target_hint=target_hint)
        if use_case == "explain":
            context = build_session_context("explain", session_state, last_user_question=user_input)
            return build_explanation_prompt(context, question=user_input, language=language)
        if use_case == "mood":
            context = build_session_context("mood", session_state, lyrics_text=user_input)
            return build_mood_prompt(context, language=language)
        raise ValueError("use_case must be 'lyric', 'refine', 'explain', or 'mood'")

    def _request_payload(self, use_case: GPTUseCase, prompt: Any, user_input: str) -> dict[str, Any]:
        return {
            "use_case": use_case,
            "model": self.client.config.model,
            "provider": self.client.config.provider,
            "messages": prompt.messages,
            "response_format": getattr(prompt, "response_format", None),
            "user_input": user_input,
            "eval_mode": self.config.eval_mode,
            "eval_seed": self.config.eval_seed if self.config.eval_mode else None,
        }

    def _call_kwargs(self, prompt: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        response_format = getattr(prompt, "response_format", None)
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _cache_allowed(self, use_case: GPTUseCase) -> bool:
        if not self.config.cache_enabled:
            return False
        if self.config.eval_mode and use_case == "refine":
            return False
        return True

    def _parse_response(self, use_case: GPTUseCase, content: str) -> Any:
        if use_case == "refine":
            return parse_refinement_plan(content)
        if use_case in ("lyric", "explain", "mood"):
            return _parse_json_or_text(content)
        raise ValueError("use_case must be 'lyric', 'refine', 'explain', or 'mood'")

    def _result_from_cached(self, use_case: GPTUseCase, cached: str, *, latency_ms: float) -> GPTPipelineResult:
        data = json.loads(cached)
        return GPTPipelineResult(
            use_case=use_case,
            content=data["content"],
            parsed=data.get("parsed"),
            cache_hit=True,
            provider=data["provider"],
            model_version=data["model_version"],
            latency_ms=latency_ms,
            prompt_token_count=data.get("prompt_token_count"),
            completion_token_count=data.get("completion_token_count"),
            eval_mode=self.config.eval_mode,
        )

    def _log_call(self, result: GPTPipelineResult) -> None:
        if self.logger is None:
            return
        fields = {
            "provider": result.provider,
            "model_version": result.model_version,
            "use_case": result.use_case,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
            "prompt_token_count": result.prompt_token_count,
            "completion_token_count": result.completion_token_count,
            "eval_mode": result.eval_mode,
        }
        try:
            self.logger.info("gpt_call_completed", **fields)
        except TypeError:
            self.logger.info("gpt_call_completed", extra=fields)


def route_request(
    use_case: GPTUseCase,
    session_state: SessionState,
    user_input: str,
    *,
    client: GPTClient | None = None,
    cache: Any | None = None,
    config: GPTPipelineConfig | None = None,
    logger: Any | None = LOGGER,
    **kwargs: Any,
) -> GPTPipelineResult:
    """Convenience wrapper for one-off routed GPT calls."""
    pipeline = GPTPipeline(client or GPTClient.from_env(), cache=cache, config=config, logger=logger)
    return pipeline.route_request(use_case, session_state, user_input, **kwargs)


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_cached_result(result: GPTPipelineResult) -> str:
    return json.dumps(
        {
            "content": result.content,
            "parsed": _jsonable(result.parsed),
            "provider": result.provider,
            "model_version": result.model_version,
            "prompt_token_count": result.prompt_token_count,
            "completion_token_count": result.completion_token_count,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _parse_json_or_text(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"text": content}


def _token_count(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if value is not None else None


def _elapsed_ms(started_at: float, finished_at: float) -> float:
    return round((finished_at - started_at) * 1000, 3)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)
