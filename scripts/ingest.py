"""Ingest CUAD into Postgres: parse, chunk, enrich, dedup, persist.

Usage:
    uv run python scripts/ingest.py                        # both corpora
    uv run python scripts/ingest.py --corpus clause_aware
    uv run python scripts/ingest.py --limit 10 --dry-run   # fast sanity pass
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.adapters.db.engine import create_session_factory, sync_engine
from clausewise.config import get_settings
from clausewise.ingestion.pipeline import CHUNKERS, run_ingestion
from clausewise.observability import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=[*CHUNKERS, "all"],
        default="all",
        help="which chunker strategy to ingest (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only first N contracts")
    parser.add_argument("--dry-run", action="store_true", help="no database writes")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    corpora = list(CHUNKERS) if args.corpus == "all" else [args.corpus]

    factory = create_session_factory(sync_engine(settings))
    started = time.perf_counter()
    stats = run_ingestion(factory, corpora=corpora, limit=args.limit, dry_run=args.dry_run)
    elapsed = time.perf_counter() - started

    print(f"\nContracts:        {stats.contracts}")
    for corpus, count in stats.per_corpus.items():
        print(f"Chunks[{corpus}]:  {count:,}")
    print(f"Duplicates marked: {stats.duplicates:,}")
    print(f"Chunks w/ labels:  {stats.enriched:,}")
    print(
        f"Rows written:      {stats.written_contracts} contracts, {stats.written_chunks:,} chunks"
    )
    print(f"Elapsed:           {elapsed:.1f}s")
    if args.dry_run:
        print("(dry run — nothing persisted)")


if __name__ == "__main__":
    main()
