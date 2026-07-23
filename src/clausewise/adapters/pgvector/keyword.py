"""KeywordIndex port implementation on Postgres full-text search.

Ranking: ts_rank_cd (cover density) — rewards query terms appearing close
together, which suits clause retrieval ("force majeure" as a phrase beats the
words scattered across a section). Queries go through websearch_to_tsquery,
which safely parses free-form user input (quotes, OR, -exclusions) and never
raises on garbage.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clausewise.domain import Chunk, ChunkMetadata, RetrievedChunk
from clausewise.domain.retrieval import RetrievalSource

_SEARCH_SQL_BASE = """
SELECT c.id, c.contract_id, c.text, c.char_start, c.char_end, c.token_count,
       c.section_path, c.clause_types, c.duplicate_of,
       k.title AS contract_title,
       ts_rank_cd(c.tsv, websearch_to_tsquery('english', :q)) AS score
FROM chunks c
JOIN contracts k ON k.id = c.contract_id
WHERE c.corpus = :corpus AND c.tsv @@ websearch_to_tsquery('english', :q)
{contract_filter}
ORDER BY score DESC
LIMIT :k
"""


class PostgresKeywordIndex:
    """Async FTS-backed KeywordIndex."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def search(
        self,
        corpus: str,
        query: str,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        params: dict[str, object] = {"q": query, "corpus": corpus, "k": k}
        if contract_ids is not None:
            if not contract_ids:
                return []
            stmt = sa.text(
                _SEARCH_SQL_BASE.format(contract_filter="AND c.contract_id IN :contract_ids")
            ).bindparams(sa.bindparam("contract_ids", expanding=True))
            params["contract_ids"] = list(contract_ids)
        else:
            stmt = sa.text(_SEARCH_SQL_BASE.format(contract_filter=""))

        async with self._sessions() as session:
            rows = (await session.execute(stmt, params)).mappings().all()

        return [
            RetrievedChunk(
                chunk=Chunk(
                    id=row["id"],
                    contract_id=row["contract_id"],
                    text=row["text"],
                    char_start=row["char_start"],
                    char_end=row["char_end"],
                    token_count=row["token_count"],
                    metadata=ChunkMetadata(
                        contract_title=row["contract_title"],
                        section_path=tuple(row["section_path"]),
                        clause_types=tuple(row["clause_types"]),
                        duplicate_of=row["duplicate_of"],
                    ),
                ),
                score=float(row["score"]),
                source=RetrievalSource.KEYWORD,
            )
            for row in rows
        ]
