"""Shared package with cross-module schemas."""

from shared.schemas import (
    ArtifactRef,
    ChordSuggestion,
    ExplanationPart,
    GenerationRequest,
    GenerationResponse,
    MelodyNote,
)

__all__ = [
    "ArtifactRef",
    "MelodyNote",
    "ChordSuggestion",
    "ExplanationPart",
    "GenerationRequest",
    "GenerationResponse",
]

