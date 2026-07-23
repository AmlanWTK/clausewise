"""Embed all un-embedded chunks (resumable: WHERE embedding IS NULL).

Usage:
    uv run --group ml python scripts/embed.py                      # everything
    uv run --group ml python scripts/embed.py --corpus clause_aware
    uv run --group ml python scripts/embed.py --batch 128
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sqlalchemy as sa

from clausewise.adapters.db.async_engine import async_engine, create_async_session_factory
from clausewise.adapters.embeddings import SentenceTransformerEmbeddingProvider
from clausewise.adapters.pgvector import PgVectorStore
from clausewise.config import get_settings
from clausewise.domain import Chunk, ChunkMetadata
from clausewise.observability import configure_logging

_FETCH_SQL = sa.text(
    "SELECT id, contract_id, text, char_start, char_end, token_count "
    "FROM chunks WHERE embedding IS NULL AND corpus = :corpus "
    "ORDER BY id LIMIT :n"
)
_COUNT_SQL = sa.text("SELECT count(*) FROM chunks WHERE embedding IS NULL AND corpus = :corpus")


async def embed_corpus(corpus: str, batch_size: int) -> None:
    settings = get_settings()
    provider = SentenceTransformerEmbeddingProvider(settings)
    factory = create_async_session_factory(async_engine(settings))
    store = PgVectorStore(factory)

    async with factory() as session:
        remaining = (await session.execute(_COUNT_SQL, {"corpus": corpus})).scalar_one()
    print(f"[{corpus}] {remaining:,} chunks to embed with {provider.model_name}")

    done = 0
    total_tokens = 0
    started = time.perf_counter()
    while True:
        async with factory() as session:
            rows = (await session.execute(_FETCH_SQL, {"corpus": corpus, "n": batch_size})).all()
        if not rows:
            break
        chunks = [
            Chunk(
                id=r.id,
                contract_id=r.contract_id,
                text=r.text,
                char_start=r.char_start,
                char_end=r.char_end,
                token_count=r.token_count,
                metadata=ChunkMetadata(contract_title=""),
            )
            for r in rows
        ]
        batch = await provider.embed_documents([c.text for c in chunks])
        await store.upsert(corpus, chunks, batch.vectors, provider.model_name)
        done += len(chunks)
        total_tokens += batch.total_tokens
        rate = done / (time.perf_counter() - started)
        print(f"[{corpus}] {done:,}/{remaining:,} ({rate:.0f} chunks/s)", flush=True)

    elapsed = time.perf_counter() - started
    print(f"[{corpus}] done: {done:,} chunks, {total_tokens:,} tokens, {elapsed:.0f}s, cost $0.00")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["clause_aware", "fixed_size", "all"], default="all")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    configure_logging(get_settings())
    corpora = ["clause_aware", "fixed_size"] if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        await embed_corpus(corpus, args.batch)


if __name__ == "__main__":
    asyncio.run(main())
