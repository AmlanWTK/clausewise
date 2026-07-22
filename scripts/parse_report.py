"""Measure parser quality across the full CUAD corpus.

Definition of Done for Checkpoint 5: >=95% of *structured-length* contracts
(>=10k chars) parse into a non-degenerate tree (>=3 sections). Shorter
documents (addenda, joint filings, signature schedules) are frequently
legitimate prose without internal structure; they fall back to whole-document
chunking by design and are reported separately, not gamed into the metric.

Usage:
    uv run python scripts/parse_report.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.domain import Section
from clausewise.ingestion.cuad import load_cuad, to_domain_contract
from clausewise.ingestion.parser import extract_defined_terms, parse_contract


def _count(sections: tuple[Section, ...]) -> int:
    return sum(1 + _count(s.children) for s in sections)


def _depth(sections: tuple[Section, ...]) -> int:
    return 1 + max((_depth(s.children) for s in sections), default=0) if sections else 0


def _covered_chars(sections: tuple[Section, ...]) -> int:
    # Top-level sections don't overlap by construction; sum their spans.
    return sum(s.char_end - s.char_start for s in sections)


def main() -> None:
    contracts = [to_domain_contract(c) for c in load_cuad()]

    section_counts: list[int] = []
    depths: list[int] = []
    coverages: list[float] = []
    term_counts: list[int] = []
    degenerate: list[str] = []

    for contract in contracts:
        tree = parse_contract(contract)
        n = _count(tree)
        section_counts.append(n)
        depths.append(_depth(tree))
        coverages.append(_covered_chars(tree) / max(len(contract.text), 1))
        term_counts.append(len(extract_defined_terms(contract.text)))
        if n < 3:
            degenerate.append(f"{contract.title} ({n} sections)")

    total = len(contracts)
    ok = total - len(degenerate)
    section_counts.sort()
    coverages_sorted = sorted(coverages)

    long_total = sum(1 for c in contracts if len(c.text) >= 10_000)
    long_degenerate = sum(
        1 for c in contracts if len(c.text) >= 10_000 and _count(parse_contract(c)) < 3
    )
    long_ok = long_total - long_degenerate

    print(f"Contracts parsed:        {total}")
    print(f"Non-degenerate overall:  {ok}  ({100 * ok / total:.1f}%)")
    print(
        f"Structured (>=10k chars): {long_ok}/{long_total}  "
        f"({100 * long_ok / long_total:.1f}%)  [DoD target: >=95%]"
    )
    print(
        "Sections per contract:   "
        f"median {section_counts[total // 2]} · "
        f"p10 {section_counts[total // 10]} · "
        f"max {section_counts[-1]}"
    )
    print(f"Max tree depth:          {max(depths)} · median {sorted(depths)[total // 2]}")
    print(
        "Text coverage by tree:   "
        f"median {100 * coverages_sorted[total // 2]:.0f}% · "
        f"p10 {100 * coverages_sorted[total // 10]:.0f}%"
    )
    print(f"Defined terms/contract:  median {sorted(term_counts)[total // 2]}")
    print(f"Mean defined terms:      {statistics.mean(term_counts):.1f}")

    if degenerate:
        print(f"\nDegenerate parses ({len(degenerate)}):")
        for title in degenerate[:15]:
            print(f"  - {title}")
        if len(degenerate) > 15:
            print(f"  ... and {len(degenerate) - 15} more")

    if long_ok / long_total < 0.95:
        print("\nFAIL: below the 95% DoD threshold (structured-length contracts).")
        raise SystemExit(1)
    print("\nPASS: DoD threshold met.")


if __name__ == "__main__":
    main()
