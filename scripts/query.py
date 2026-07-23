"""Ad-hoc dense retrieval query — the first end-to-end search.

Usage:
    uv run --group ml python scripts/query.py "What is the governing law?"
    uv run --group ml python scripts/query.py -c fixed_size -k 5 "audit rights"
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.adapters.db.async_engine import async_engine, create_async_session_factory
from clausewise.adapters.embeddings import SentenceTransformerEmbeddingProvider
from clausewise.adapters.pgvector import PgVectorStore
from clausewise.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-c", "--corpus", default="clause_aware")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    provider = SentenceTransformerEmbeddingProvider(settings)
    store = PgVectorStore(create_async_session_factory(async_engine(settings)))

    query_vector = await provider.embed_query(args.question)
    results = await store.search(args.corpus, query_vector, k=args.k)

    print(f"\nQ: {args.question}   [corpus={args.corpus}]\n" + "=" * 70)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        path = " > ".join(chunk.metadata.section_path) or "(no section)"
        dup = f"  [dup of {chunk.metadata.duplicate_of}]" if chunk.metadata.duplicate_of else ""
        print(f"\n#{rank}  score={result.score:.4f}  {chunk.metadata.contract_title[:50]}")
        print(f"    {path}{dup}")
        text = chunk.text[:400].replace("\n", " ")
        print(f"    {text}{'...' if len(chunk.text) > 400 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
