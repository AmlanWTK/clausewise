"""Integration test for PostgresKeywordIndex — real FTS, fixture rows."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
import sqlalchemy as sa

from clausewise.adapters.db.async_engine import async_engine, create_async_session_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from clausewise.adapters.pgvector.keyword import PostgresKeywordIndex
from clausewise.config import get_settings

pytestmark = pytest.mark.integration

CORPUS = "clause_aware"


@pytest_asyncio.fixture()
async def index() -> AsyncIterator[PostgresKeywordIndex]:
    settings = get_settings()
    engine = async_engine(settings)
    factory: async_sessionmaker[AsyncSession] = create_async_session_factory(engine)
    try:
        async with factory() as session:
            await session.execute(sa.text("SELECT tsv FROM chunks LIMIT 1"))
    except Exception:
        pytest.skip("Postgres not ready — compose up + alembic upgrade head")

    async with factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO contracts (id, title, text) VALUES ('kw_contract', 'KwTest', 'x') "
                "ON CONFLICT (id) DO NOTHING"
            )
        )
        await session.execute(
            sa.text(
                "INSERT INTO chunks (id, corpus, contract_id, text, char_start, char_end, "
                "token_count) VALUES "
                "('kw_c1', :corpus, 'kw_contract', "
                "'Neither party shall be liable for delays caused by force majeure events "
                "including acts of God.', 0, 99, 20), "
                "('kw_c2', :corpus, 'kw_contract', "
                "'Payment is due within thirty days of the invoice date.', 0, 55, 11) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"corpus": CORPUS},
        )
        await session.commit()

    yield PostgresKeywordIndex(factory)

    async with factory() as session:
        await session.execute(sa.text("DELETE FROM chunks WHERE id LIKE 'kw_%'"))
        await session.execute(sa.text("DELETE FROM contracts WHERE id = 'kw_contract'"))
        await session.commit()
    await engine.dispose()


@pytest.mark.asyncio
async def test_exact_phrase_ranks_first(index: PostgresKeywordIndex) -> None:
    # Scope to the fixture contract — the shared corpus contains real CUAD
    # force-majeure clauses that would out-rank the fixture row.
    results = await index.search(CORPUS, "force majeure", k=5, contract_ids=["kw_contract"])
    assert results, "expected FTS hits"
    assert results[0].chunk.id == "kw_c1"


@pytest.mark.asyncio
async def test_no_match_and_garbage_are_safe(index: PostgresKeywordIndex) -> None:
    assert await index.search(CORPUS, "xyzzy plugh nonexistentterm", k=5) == []
    # websearch_to_tsquery must not raise on odd syntax
    assert isinstance(await index.search(CORPUS, '"unbalanced AND -', k=5), list)


@pytest.mark.asyncio
async def test_empty_query_returns_empty(index: PostgresKeywordIndex) -> None:
    assert await index.search(CORPUS, "   ", k=5) == []
