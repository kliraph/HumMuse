"""OpenAI-compatible GPT provider layer."""

from ml.gpt.abc_serializer import ABCSerializationOptions, midi_to_abc_pitch, session_state_to_abc
from ml.gpt.client import GPTClient, GPTClientError, LLMConfig, LLMConfigError, LLMResponse
from ml.gpt.context_builder import GPTUseCase, SessionContext, SessionContextBuilder
from ml.gpt.tier1_formatter import format_chord_theory_from_progression, format_tier1_summary
from ml.gpt.syllable import (
    LyricRetryResult,
    SyllableLineResult,
    SyllableValidationResult,
    count_syllables,
    generate_with_syllable_retries,
    validate_syllable_targets,
)
from ml.gpt.prompts.refine import (
    RefinementPlanValidationError,
    build_refinement_correction_message,
    generate_refinement_plan_with_retries,
    parse_refinement_plan,
)
from ml.gpt.prompts.explain import (
    ExplanationPrompt,
    build_explanation_prompt,
)
from ml.gpt.pipeline import GPTPipeline, GPTPipelineConfig, GPTPipelineResult, ResponseCache, route_request

__all__ = [
    "ABCSerializationOptions",
    "GPTClient",
    "GPTClientError",
    "GPTUseCase",
    "GPTPipeline",
    "GPTPipelineConfig",
    "GPTPipelineResult",
    "LLMConfig",
    "LLMConfigError",
    "LLMResponse",
    "LyricRetryResult",
    "SessionContext",
    "SessionContextBuilder",
    "SyllableLineResult",
    "SyllableValidationResult",
    "RefinementPlanValidationError",
    "ResponseCache",
    "ExplanationPrompt",
    "build_explanation_prompt",
    "build_refinement_correction_message",
    "count_syllables",
    "format_chord_theory_from_progression",
    "format_tier1_summary",
    "generate_refinement_plan_with_retries",
    "midi_to_abc_pitch",
    "parse_refinement_plan",
    "route_request",
    "generate_with_syllable_retries",
    "session_state_to_abc",
    "validate_syllable_targets",
]
