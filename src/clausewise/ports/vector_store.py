"""Vector store port."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clausewise.domain import Chunk, RetrievedChunk, Vector


@runtime_checkable
class VectorStore(Protocol):
    """Persists chunks with embeddings and answers k-NN queries.

    Implementations: pgvector (default), in-memory fake for tests.
    ``corpus`` identifies a logical index (e.g. "clause_aware" vs
    "fixed_size") so ablation variants coexist in one store.
    """

    async def upsert(
        self,
        corpus: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        embedding_model: str,
    ) -> int:
        """Idempotently store chunks + vectors. Returns number written."""
        ...

    async def search(
        self,
        corpus: str,
        query_vector: Vector,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k by cosine similarity, optionally scoped to contracts.

        Filtering happens inside the store (pre-/in-search), never as a
        post-filter — a post-filter silently returns fewer than k results.
        """
        ...


@runtime_checkable
class KeywordIndexWriter(Protocol):
    """Optional write-side companion for keyword indexing (see keyword_index)."""

    async def index(self, corpus: str, chunks: Sequence[Chunk]) -> int: ...
