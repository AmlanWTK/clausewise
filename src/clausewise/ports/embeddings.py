"""Embedding provider port."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clausewise.domain import EmbeddingBatch, Vector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors.

    Implementations: local sentence-transformers (default, $0), OpenAI, etc.
    Raise ``ProviderError`` on failure after internal retries.
    """

    @property
    def model_name(self) -> str:
        """Identifier persisted alongside vectors (re-embedding is a migration)."""
        ...

    @property
    def dimensions(self) -> int:
        """Output vector width; must match the DB column definition."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed chunk texts for indexing. Implementations batch internally."""
        ...

    async def embed_query(self, text: str) -> Vector:
        """Embed a user query. May apply model-specific query prefixes."""
        ...
