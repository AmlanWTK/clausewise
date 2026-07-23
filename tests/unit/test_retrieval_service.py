"""Unit tests for RetrievalService — fakes only, no I/O."""

import pytest

from clausewise.domain import Chunk, ChunkMetadata
from clausewise.domain.retrieval import RetrievalSource
from clausewise.retrieval import RetrievalMode, RetrievalService
from tests.fakes import FakeEmbeddingProvider, InMemoryKeywordIndex, InMemoryVectorStore


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        contract_id="k1",
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        metadata=ChunkMetadata(contract_title="T"),
    )


async def _service_with_corpus() -> RetrievalService:
    provider = FakeEmbeddingProvider()
    vectors = InMemoryVectorStore()
    keywords = InMemoryKeywordIndex()
    chunks = [
        _chunk("c1", "payment obligations net thirty days"),
        _chunk("c2", "force majeure excuses performance"),
        _chunk("c3", "governing law delaware courts"),
    ]
    batch = await provider.embed_documents([c.text for c in chunks])
    await vectors.upsert("test", chunks, batch.vectors, provider.model_name)
    await keywords.index("test", chunks)
    return RetrievalService(provider, vectors, keywords)


@pytest.mark.asyncio
async def test_dense_mode_returns_dense_results() -> None:
    service = await _service_with_corpus()
    results = await service.retrieve(
        "force majeure excuses performance", corpus="test", mode=RetrievalMode.DENSE, k=2
    )
    assert results[0].chunk.id == "c2"
    assert all(r.source is RetrievalSource.DENSE for r in results)


@pytest.mark.asyncio
async def test_keyword_mode_returns_keyword_results() -> None:
    service = await _service_with_corpus()
    results = await service.retrieve(
        "force majeure", corpus="test", mode=RetrievalMode.KEYWORD, k=2
    )
    assert results[0].chunk.id == "c2"
    assert all(r.source is RetrievalSource.KEYWORD for r in results)


@pytest.mark.asyncio
async def test_hybrid_fuses_and_tags_source() -> None:
    service = await _service_with_corpus()
    results = await service.retrieve("force majeure", corpus="test", mode=RetrievalMode.HYBRID, k=3)
    assert results
    assert all(r.source is RetrievalSource.FUSED for r in results)
    # exact-term chunk should be top: keyword ranks it #1, dense hash-embedding
    # ranks it somewhere — RRF agreement puts it first.
    assert results[0].chunk.id == "c2"


@pytest.mark.asyncio
async def test_hybrid_survives_empty_keyword_results() -> None:
    service = await _service_with_corpus()
    results = await service.retrieve("zzz qqq www", corpus="test", mode=RetrievalMode.HYBRID, k=2)
    # keyword finds nothing; dense still returns nearest neighbors
    assert len(results) == 2
