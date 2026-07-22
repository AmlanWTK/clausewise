"""Unit tests for both chunkers."""

from itertools import pairwise

import pytest

from clausewise.domain import Chunk, Contract
from clausewise.ingestion.chunkers import ChunkerConfig, ClauseAwareChunker, FixedSizeChunker
from clausewise.ingestion.parser import parse_contract
from clausewise.ports import Chunker

CONTRACT_TEXT = """PREAMBLE: the parties enter this agreement freely.

ARTICLE I - DEFINITIONS

1.1 Terms. Words have their plain meaning unless defined otherwise in this section.

1.2 Interpretation. Headings are for convenience only and shall not affect construction.

ARTICLE II - OBLIGATIONS

2.1 Delivery. Seller shall deliver the goods to Buyer's warehouse within thirty days.

2.2 Payment. Buyer shall pay within sixty days of receipt of a conforming invoice.
"""


def _contract() -> Contract:
    return Contract(id="k1", title="Test MSA", text=CONTRACT_TEXT, parties=("Seller", "Buyer"))


# Fixture contracts are tiny; scale min_tokens down so merge behavior mirrors
# real-corpus proportions instead of collapsing everything into one chunk.
_TEST_CONFIG = ChunkerConfig(min_tokens=8, target_tokens=350, max_tokens=480)


def _chunks(chunker: Chunker | None = None) -> list[Chunk]:
    contract = _contract()
    chunker = chunker or ClauseAwareChunker(_TEST_CONFIG)
    return chunker.chunk(contract, parse_contract(contract))


class TestClauseAware:
    def test_offsets_point_at_canonical_text(self) -> None:
        contract = _contract()
        for chunk in _chunks():
            source = contract.text[chunk.char_start : chunk.char_end]
            # chunk.text = optional "[path] " prefix + exact source span
            assert chunk.text.endswith(source)

    def test_section_paths_recorded(self) -> None:
        paths = {c.metadata.section_path for c in _chunks()}
        assert any("ARTICLE II" in p for p in paths if p)

    def test_no_overlapping_spans(self) -> None:
        spans = sorted((c.char_start, c.char_end) for c in _chunks())
        for (_, prev_end), (next_start, _) in pairwise(spans):
            assert prev_end <= next_start

    def test_all_substantive_text_is_covered(self) -> None:
        contract = _contract()
        covered = bytearray(len(contract.text))
        for c in _chunks():
            for i in range(c.char_start, c.char_end):
                covered[i] = 1
        missing = "".join(contract.text[i] for i in range(len(contract.text)) if not covered[i])
        assert not missing.strip(), f"Uncovered text: {missing[:100]!r}"

    def test_tiny_units_are_merged(self) -> None:
        config = ChunkerConfig(min_tokens=40, target_tokens=350, max_tokens=480)
        chunks = ClauseAwareChunker(config).chunk(_contract(), parse_contract(_contract()))
        # No chunk should be under min_tokens unless it's isolated (non-adjacent).
        tiny = [c for c in chunks if c.token_count < 10]
        assert not tiny

    def test_oversized_sections_are_split(self) -> None:
        # One giant unstructured "section": 400 sentences.
        text = "SECTION 1 Everything\n\n" + " ".join(
            f"Sentence number {i} says something about obligations." for i in range(400)
        )
        contract = Contract(id="k2", title="Big", text=text)
        config = ChunkerConfig(target_tokens=100, min_tokens=10, max_tokens=120)
        chunks = ClauseAwareChunker(config).chunk(contract, parse_contract(contract))
        assert len(chunks) > 5
        assert all(c.token_count <= 130 for c in chunks)  # target + one sentence slack

    def test_unparsed_contract_still_chunks(self) -> None:
        contract = Contract(id="k3", title="Prose", text="Just words. " * 50)
        chunks = ClauseAwareChunker().chunk(contract, ())
        assert chunks
        assert chunks[0].metadata.section_path == ()


class TestFixedSize:
    def test_offsets_are_exact_slices(self) -> None:
        contract = _contract()
        for chunk in _chunks(FixedSizeChunker()):
            assert chunk.text == contract.text[chunk.char_start : chunk.char_end]

    def test_windows_overlap(self) -> None:
        text = "word " * 2000
        contract = Contract(id="k4", title="Long", text=text)
        config = ChunkerConfig(target_tokens=100, min_tokens=10, max_tokens=120, overlap_tokens=20)
        chunks = FixedSizeChunker(config).chunk(contract, ())
        assert len(chunks) > 2
        for a, b in pairwise(chunks):
            assert b.char_start < a.char_end  # consecutive windows share text

    def test_covers_whole_document(self) -> None:
        contract = _contract()
        chunks = _chunks(FixedSizeChunker())
        assert chunks[0].char_start == contract.text.index("PREAMBLE")
        assert chunks[-1].char_end >= len(contract.text.rstrip()) - 1

    def test_ignores_structure(self) -> None:
        # Baseline never records section paths — that's the point.
        assert all(c.metadata.section_path == () for c in _chunks(FixedSizeChunker()))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        ChunkerConfig(min_tokens=500, target_tokens=100, max_tokens=50)


def test_ids_are_deterministic_and_strategy_scoped() -> None:
    clause = _chunks()
    clause_again = _chunks()
    fixed = _chunks(FixedSizeChunker())
    assert [c.id for c in clause] == [c.id for c in clause_again]
    assert {c.id for c in clause}.isdisjoint({c.id for c in fixed})
