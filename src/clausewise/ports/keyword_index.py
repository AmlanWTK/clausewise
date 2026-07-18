"""Keyword (lexical) search port."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clausewise.domain import RetrievedChunk


@runtime_checkable
class KeywordIndex(Protocol):
    """Lexical search over chunk text (BM25-style ranking).

    Exists because legal Q&A needs exact term hits ("force majeure",
    "indemnification") that dense embeddings can blur.
    Implementation: Postgres full-text search (Checkpoint 10).
    """

    async def search(
        self,
        corpus: str,
        query: str,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...
