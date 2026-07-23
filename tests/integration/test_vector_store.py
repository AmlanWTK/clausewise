"""Integration tests for PgVectorStore — real Postgres, hand-made vectors.

Deliberately does NOT require sentence-transformers: the store's contract
(upsert, ranked search, filtering) is independent of any model.
"""

import math
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clausewise.adapters.db.async_engine import async_engine, create_async_session_factory
from clausewise.adapters.pgvector import PgVectorStore
from clausewise.config import get_settings
from clausewise.domain import Chunk, ChunkMetadata, Vector

pytestmark = pytest.mark.integration

DIMS = 384
CORPUS = "clause_aware"  # reuse existing corpus value; rows are test-scoped


def _unit_vector(hot: int) -> Vector:
    raw = [0.001] * DIMS
    raw[hot] = 1.0
    norm = math.sqrt(sum(x * x for x in raw))
    return tuple(x / norm for x in raw)


def _chunk(chunk_id: str, contract_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        contract_id=contract_id,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=5,
        metadata=ChunkMetadata(contract_title="VecTest"),
    )


@pytest_asyncio.fixture()
async def store_and_factory() -> (
    AsyncIterator[tuple[PgVectorStore, async_sessionmaker[AsyncSession]]]
):
    settings = get_settings()
    engine = async_engine(settings)
    factory = create_async_session_factory(engine)
    try:
        async with factory() as session:
            await session.execute(sa.text("SELECT embedding FROM chunks LIMIT 1"))
    except Exception:
        pytest.skip("Postgres not ready — compose up + alembic upgrade head")

    async with factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO contracts (id, title, text) VALUES "
                "('vt_contract_1', 'VecTest', 'x'), ('vt_contract_2', 'VecTest', 'y') "
                "ON CONFLICT (id) DO NOTHING"
            )
        )
        await session.execute(
            sa.text(
                "INSERT INTO chunks (id, corpus, contract_id, text, char_start, char_end, "
                "token_count) VALUES "
                "('vt_c1', :corpus, 'vt_contract_1', 'alpha clause', 0, 12, 3), "
                "('vt_c2', :corpus, 'vt_contract_1', 'beta clause', 0, 11, 3), "
                "('vt_c3', :corpus, 'vt_contract_2', 'gamma clause', 0, 12, 3) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"corpus": CORPUS},
        )
        await session.commit()

    yield PgVectorStore(factory), factory

    async with factory() as session:
        await session.execute(sa.text("DELETE FROM chunks WHERE id LIKE 'vt_%'"))
        await session.execute(sa.text("DELETE FROM contracts WHERE id LIKE 'vt_%'"))
        await session.commit()
    await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_then_search_ranks_by_similarity(
    store_and_factory: tuple[PgVectorStore, async_sessionmaker[AsyncSession]],
) -> None:
    store, _ = store_and_factory
    chunks = [
        _chunk("vt_c1", "vt_contract_1", "alpha clause"),
        _chunk("vt_c2", "vt_contract_1", "beta clause"),
        _chunk("vt_c3", "vt_contract_2", "gamma clause"),
    ]
    vectors = [_unit_vector(0), _unit_vector(100), _unit_vector(200)]
    written = await store.upsert(CORPUS, chunks, vectors, "test-model")
    assert written == 3

    # Scope to fixture contracts: the corpus also contains the real embedded
    # CUAD chunks, which must not leak into this test's ranking.
    vt = await store.search(
        CORPUS,
        _unit_vector(100),
        k=3,
        contract_ids=["vt_contract_1", "vt_contract_2"],
    )
    assert vt[0].chunk.id == "vt_c2"
    assert vt[0].score == pytest.approx(1.0, abs=1e-4)
    assert vt[0].score > vt[1].score


@pytest.mark.asyncio
async def test_contract_filter_is_applied_in_sql(
    store_and_factory: tuple[PgVectorStore, async_sessionmaker[AsyncSession]],
) -> None:
    store, _ = store_and_factory
    chunks = [
        _chunk("vt_c1", "vt_contract_1", "alpha clause"),
        _chunk("vt_c3", "vt_contract_2", "gamma clause"),
    ]
    await store.upsert(CORPUS, chunks, [_unit_vector(0), _unit_vector(200)], "test-model")

    results = await store.search(CORPUS, _unit_vector(0), k=10, contract_ids=["vt_contract_2"])
    vt = [r for r in results if r.chunk.id.startswith("vt_")]
    assert [r.chunk.id for r in vt] == ["vt_c3"]

    assert await store.search(CORPUS, _unit_vector(0), k=5, contract_ids=[]) == []
