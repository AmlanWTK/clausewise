"""Run both chunkers over full CUAD and report size distributions.

Definition of Done for Checkpoint 6: both chunkers produce valid chunk sets
over the whole corpus, with token histograms inside configured bounds.

Usage:
    uv run python scripts/chunk_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.ingestion.chunkers import ClauseAwareChunker, FixedSizeChunker
from clausewise.ingestion.cuad import load_cuad, to_domain_contract
from clausewise.ingestion.parser import parse_contract
from clausewise.ports import Chunker


def report(name: str, chunker: Chunker) -> None:
    token_counts: list[int] = []
    chunk_totals: list[int] = []
    empty_contracts = 0

    for cuad in load_cuad():
        contract = to_domain_contract(cuad)
        chunks = chunker.chunk(contract, parse_contract(contract))
        if not chunks:
            empty_contracts += 1
            continue
        chunk_totals.append(len(chunks))
        token_counts.extend(c.token_count for c in chunks)

    token_counts.sort()
    n = len(token_counts)

    def pct(q: float) -> int:
        return token_counts[min(int(q * n), n - 1)]

    print(f"=== {name}")
    print(f"  chunks total:        {n:,}")
    print(f"  chunks/contract:     median {sorted(chunk_totals)[len(chunk_totals) // 2]}")
    print(
        f"  tokens/chunk:        p10 {pct(0.10)} · p50 {pct(0.50)} · "
        f"p90 {pct(0.90)} · max {token_counts[-1]}"
    )
    print(f"  contracts w/o chunks: {empty_contracts}")
    over = sum(1 for t in token_counts if t > 550)
    print(f"  chunks over 550 est. tokens: {over} ({100 * over / n:.2f}%)")
    print()


def main() -> None:
    report("clause_aware", ClauseAwareChunker())
    report("fixed_size", FixedSizeChunker())


if __name__ == "__main__":
    main()
