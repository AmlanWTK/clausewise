"""Inspect how the parser sees a specific contract — for iterating on rules.

Usage:
    uv run python scripts/debug_parse.py --list                 # degenerate titles
    uv run python scripts/debug_parse.py --title BANGIINC       # show one contract
    uv run python scripts/debug_parse.py --title BANGIINC -n 120
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.domain import Section
from clausewise.ingestion.cuad import load_cuad, to_domain_contract
from clausewise.ingestion.parser import parse_contract


def _count(sections: tuple[Section, ...]) -> int:
    return sum(1 + _count(s.children) for s in sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list degenerate contracts")
    parser.add_argument("--title", help="substring of a contract title to inspect")
    parser.add_argument("-n", type=int, default=80, help="lines of text to show")
    args = parser.parse_args()

    contracts = [to_domain_contract(c) for c in load_cuad()]

    if args.list:
        for contract in contracts:
            n = _count(parse_contract(contract))
            if n < 3:
                print(f"{n:3d}  {contract.title}")
        return

    if not args.title:
        raise SystemExit("Provide --list or --title <substring>")

    matches = [c for c in contracts if args.title.lower() in c.title.lower()]
    if not matches:
        raise SystemExit(f"No contract title contains {args.title!r}")
    contract = matches[0]
    tree = parse_contract(contract)

    print(f"=== {contract.title}")
    print(f"=== length {len(contract.text):,} chars · {_count(tree)} sections parsed\n")

    print("--- first lines (| marks line start) ---")
    for i, line in enumerate(contract.text.splitlines()[: args.n], start=1):
        shown = line[:110] + ("..." if len(line) > 110 else "")
        print(f"{i:4d} |{shown}")

    def show(sections: tuple[Section, ...], indent: int = 0) -> None:
        for s in sections:
            print(f"{'  ' * indent}{s.number!r}  {s.heading[:60]!r}  [{s.char_start}:{s.char_end}]")
            show(s.children, indent + 1)

    print("\n--- parsed tree ---")
    show(tree)


if __name__ == "__main__":
    main()
