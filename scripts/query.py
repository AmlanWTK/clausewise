"""Ad-hoc retrieval query with mode switch (dense | keyword | hybrid).

Usage:
    uv run --group ml python scripts/query.py "What is the governing law?"
    uv run --group ml python scripts/query.py -m dense "audit rights"
    uv run --group ml python scripts/query.py -m keyword "force majeure"
    uv run --group ml python scripts/query.py -c fixed_size -k 5 "termination"
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.adapters.db.async_engine import async_engine, create_async_session_factory
from clausewise.adapters.embeddings import SentenceTransformerEmbeddingProvider
from clausewise.adapters.pgvector import PgVectorStore
from clausewise.adapters.pgvector.keyword import PostgresKeywordIndex
from clausewise.config import get_settings
from clausewise.retrieval import RetrievalMode, RetrievalService


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-c", "--corpus", default="clause_aware")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-m", "--mode", choices=[m.value for m in RetrievalMode], default="hybrid")
    args = parser.parse_args()

    settings = get_settings()
    factory = create_async_session_factory(async_engine(settings))
    service = RetrievalService(
        embeddings=SentenceTransformerEmbeddingProvider(settings),
        vector_store=PgVectorStore(factory),
        keyword_index=PostgresKeywordIndex(factory),
    )

    results = await service.retrieve(
        args.question, corpus=args.corpus, mode=RetrievalMode(args.mode), k=args.k
    )

    print(f"\nQ: {args.question}   [corpus={args.corpus} mode={args.mode}]\n" + "=" * 70)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        path = " > ".join(chunk.metadata.section_path) or "(no section)"
        dup = f"  [dup of {chunk.metadata.duplicate_of}]" if chunk.metadata.duplicate_of else ""
        print(
            f"\n#{rank}  score={result.score:.4f} ({result.source})  "
            f"{chunk.metadata.contract_title[:50]}"
        )
        print(f"    {path}{dup}")
        text = chunk.text[:400].replace("\n", " ")
        print(f"    {text}{'...' if len(chunk.text) > 400 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
