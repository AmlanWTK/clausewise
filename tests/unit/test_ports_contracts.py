"""Contract tests: fakes must genuinely satisfy the port Protocols.

If a fake drifts from a port's signature, these tests (and mypy) fail —
meaning every service tested against fakes is tested against the real
interface shape.
"""

import pytest

from clausewise.domain import Chunk, ChunkMetadata, Contract, RetrievedChunk, Section
from clausewise.domain.retrieval import RetrievalSource
from clausewise.ports import (
    Chunker,
    EmbeddingProvider,
    KeywordIndex,
    LLMProvider,
    Reranker,
    VectorStore,
)
from tests.fakes import (
    FakeChunker,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeReranker,
    InMemoryKeywordIndex,
    InMemoryVectorStore,
)


def _chunk(chunk_id: str, text: str, contract_id: str = "k1") -> Chunk:
    return Chunk(
        id=chunk_id,
        contract_id=contract_id,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        metadata=ChunkMetadata(contract_title="T"),
    )


def _dense(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, source=RetrievalSource.DENSE)


def test_fakes_satisfy_protocols() -> None:
    # runtime_checkable verifies method presence; mypy verifies full signatures.
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(FakeLLMProvider(), LLMProvider)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(InMemoryKeywordIndex(), KeywordIndex)
    assert isinstance(FakeChunker(), Chunker)


@pytest.mark.asyncio
async def test_embedding_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dimensions=16)
    batch = await provider.embed_documents(["indemnification clause", "termination clause"])
    assert batch.dimensions == 16
    assert len(batch.vectors) == 2
    again = await provider.embed_documents(["indemnification clause"])
    assert batch.vectors[0] == again.vectors[0]


@pytest.mark.asyncio
async def test_vector_store_ranks_identical_text_first() -> None:
    provider = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    chunks = [_chunk("c1", "governing law of New York"), _chunk("c2", "payment terms net 30")]
    batch = await provider.embed_documents([c.text for c in chunks])
    await store.upsert("test", chunks, batch.vectors, provider.model_name)

    results = await store.search("test", await provider.embed_query("governing law of New York"))
    assert results[0].chunk.id == "c1"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].source is RetrievalSource.DENSE


@pytest.mark.asyncio
async def test_vector_store_contract_filter_scopes_results() -> None:
    provider = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    chunks = [_chunk("c1", "alpha", contract_id="k1"), _chunk("c2", "alpha", contract_id="k2")]
    batch = await provider.embed_documents([c.text for c in chunks])
    await store.upsert("test", chunks, batch.vectors, provider.model_name)

    results = await store.search("test", await provider.embed_query("alpha"), contract_ids=["k2"])
    assert [r.chunk.contract_id for r in results] == ["k2"]


@pytest.mark.asyncio
async def test_keyword_index_finds_exact_terms() -> None:
    index = InMemoryKeywordIndex()
    await index.index("test", [_chunk("c1", "force majeure event"), _chunk("c2", "payment terms")])
    results = await index.search("test", "force majeure", k=5)
    assert results and results[0].chunk.id == "c1"


@pytest.mark.asyncio
async def test_reranker_reorders_and_truncates() -> None:
    reranker = FakeReranker()
    # Dense retrieval put the wrong chunk first; the reranker should fix it.
    candidates = [
        _dense(_chunk("c1", "payment terms"), 0.9),
        _dense(_chunk("c2", "force majeure clause"), 0.8),
        _dense(_chunk("c3", "miscellaneous provisions"), 0.7),
    ]
    reranked = await reranker.rerank("force majeure", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk.id == "c2"
    assert all(r.source is RetrievalSource.RERANKED for r in reranked)


def test_chunker_preserves_offsets() -> None:
    contract = Contract(id="k1", title="T", text="AAAA BBBB CCCC")
    sections = [
        Section(number="1", heading="A", level=0, char_start=0, char_end=4),
        Section(number="2", heading="B", level=0, char_start=5, char_end=9),
    ]
    chunks = FakeChunker().chunk(contract, sections)
    assert [contract.text[c.char_start : c.char_end] for c in chunks] == ["AAAA", "BBBB"]
