"""Mode-switched retrieval service — depends only on ports.

The mode enum is ablation axis #2 (dense vs hybrid); the eval harness sweeps
it as configuration, never as a code change. Hybrid retrieves candidate_k
from each source before fusing: fuse wide, return narrow.
"""

import asyncio
from collections.abc import Sequence
from enum import StrEnum

from clausewise.domain import RetrievedChunk
from clausewise.ports import EmbeddingProvider, KeywordIndex, VectorStore
from clausewise.retrieval.fusion import reciprocal_rank_fusion


class RetrievalMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RetrievalService:
    """First-stage retrieval: dense, keyword, or RRF-fused hybrid."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vector_store
        self._keywords = keyword_index

    async def retrieve(
        self,
        question: str,
        *,
        corpus: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        k: int = 10,
        candidate_k: int = 50,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if mode is RetrievalMode.DENSE:
            return await self._dense(question, corpus, k, contract_ids)
        if mode is RetrievalMode.KEYWORD:
            return await self._keywords.search(corpus, question, k=k, contract_ids=contract_ids)
        dense_results, keyword_results = await asyncio.gather(
            self._dense(question, corpus, candidate_k, contract_ids),
            self._keywords.search(corpus, question, k=candidate_k, contract_ids=contract_ids),
        )
        return reciprocal_rank_fusion([dense_results, keyword_results], k=k)

    async def _dense(
        self,
        question: str,
        corpus: str,
        k: int,
        contract_ids: Sequence[str] | None,
    ) -> list[RetrievedChunk]:
        query_vector = await self._embeddings.embed_query(question)
        return await self._vectors.search(corpus, query_vector, k=k, contract_ids=contract_ids)
