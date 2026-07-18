"""Domain layer: pure business objects with zero external dependencies.

Rules for this package (enforced in review, verified by import-linter later):
- No imports from adapters, ports, API, or any third-party SDK.
- All types are immutable (frozen dataclasses) — state changes produce new objects.
"""

from clausewise.domain.documents import Chunk, ChunkMetadata, Contract, Section
from clausewise.domain.errors import (
    ClausewiseError,
    ConfigurationError,
    GenerationError,
    IngestionError,
    ProviderError,
    RetrievalError,
)
from clausewise.domain.generation import Answer, Citation, LLMResponse, Refusal, TokenUsage
from clausewise.domain.retrieval import EmbeddingBatch, RetrievedChunk, Vector

__all__ = [
    "Answer",
    "Chunk",
    "ChunkMetadata",
    "Citation",
    "ClausewiseError",
    "ConfigurationError",
    "Contract",
    "EmbeddingBatch",
    "GenerationError",
    "IngestionError",
    "LLMResponse",
    "ProviderError",
    "Refusal",
    "RetrievalError",
    "RetrievedChunk",
    "Section",
    "TokenUsage",
    "Vector",
]
