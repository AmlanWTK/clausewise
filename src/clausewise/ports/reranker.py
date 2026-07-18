"""Re-ranker port."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clausewise.domain import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Re-scores retrieved candidates against the query with a stronger model.

    Pattern: retrieve wide (e.g. top-50), rerank, keep narrow (top-k).
    Implementations: local cross-encoder (default, $0), API rerankers.
    Returned scores use source=RERANKED and are not comparable to input scores.
    """

    @property
    def model_name(self) -> str: ...

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int = 10,
    ) -> list[RetrievedChunk]: ...
