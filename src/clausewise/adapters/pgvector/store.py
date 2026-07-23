"""VectorStore port implementation on pgvector.

Vectors travel as pgvector text literals ("[0.1,0.2,...]") with explicit
casts — no driver-specific type registration needed, works identically with
asyncpg locally and any pooler in production.

Note on "upsert": chunk rows already exist (ingestion owns them); this store
attaches embeddings to them. Chunks and vectors live in ONE table, so k-NN
search with metadata filters is a single indexed query — that's the point of
choosing pgvector.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clausewise.domain import Chunk, ChunkMetadata, RetrievedChunk, Vector
from clausewise.domain.errors import RetrievalError
from clausewise.domain.retrieval import RetrievalSource


def _to_literal(vector: Vector) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


_UPDATE_SQL = sa.text(
    "UPDATE chunks SET embedding = CAST(:vec AS vector), embedding_model = :model WHERE id = :id"
)

_SEARCH_SQL_BASE = """
SELECT c.id, c.contract_id, c.text, c.char_start, c.char_end, c.token_count,
       c.section_path, c.clause_types, c.duplicate_of,
       k.title AS contract_title,
       1 - (c.embedding <=> CAST(:qv AS vector)) AS score
FROM chunks c
JOIN contracts k ON k.id = c.contract_id
WHERE c.corpus = :corpus AND c.embedding IS NOT NULL
{contract_filter}
ORDER BY c.embedding <=> CAST(:qv AS vector)
LIMIT :k
"""


class PgVectorStore:
    """Async pgvector-backed VectorStore."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def upsert(
        self,
        corpus: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        embedding_model: str,
    ) -> int:
        del corpus  # chunk ids are globally unique; corpus already on the row
        if len(chunks) != len(vectors):
            msg = f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
            raise RetrievalError(msg)
        params = [
            {"id": chunk.id, "vec": _to_literal(vector), "model": embedding_model}
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        async with self._sessions() as session:
            await session.execute(_UPDATE_SQL, params)
            await session.commit()
        return len(params)

    async def search(
        self,
        corpus: str,
        query_vector: Vector,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        contract_filter = ""
        params: dict[str, object] = {
            "qv": _to_literal(query_vector),
            "corpus": corpus,
            "k": k,
        }
        stmt = None
        if contract_ids is not None:
            if not contract_ids:
                return []
            contract_filter = "AND c.contract_id IN :contract_ids"
            stmt = sa.text(_SEARCH_SQL_BASE.format(contract_filter=contract_filter)).bindparams(
                sa.bindparam("contract_ids", expanding=True)
            )
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
                source=RetrievalSource.DENSE,
            )
            for row in rows
        ]
