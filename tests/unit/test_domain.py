"""Unit tests for domain invariants."""

import pytest

from clausewise.domain import Chunk, ChunkMetadata, EmbeddingBatch, Refusal, Section, TokenUsage


def _chunk(char_start: int = 0, char_end: int = 10, text: str = "some text") -> Chunk:
    return Chunk(
        id="c1",
        contract_id="k1",
        text=text,
        char_start=char_start,
        char_end=char_end,
        token_count=2,
        metadata=ChunkMetadata(contract_title="T"),
    )


class TestChunk:
    def test_rejects_inverted_offsets(self) -> None:
        with pytest.raises(ValueError, match="char_end"):
            _chunk(char_start=10, char_end=5)

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="empty text"):
            _chunk(text="")

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (0, 5, True),  # overlaps start
            (5, 15, True),  # overlaps end
            (3, 7, True),  # contained
            (10, 20, False),  # adjacent (half-open: no overlap)
            (20, 30, False),  # disjoint
        ],
    )
    def test_overlaps_half_open_semantics(self, start: int, end: int, expected: bool) -> None:
        assert _chunk(0, 10).overlaps(start, end) is expected

    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            _chunk().text = "changed"  # type: ignore[misc]


class TestSection:
    def test_rejects_inverted_offsets(self) -> None:
        with pytest.raises(ValueError, match="char_end"):
            Section(number="1", heading="H", level=0, char_start=9, char_end=3)


class TestEmbeddingBatch:
    def test_rejects_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="dimensions=3"):
            EmbeddingBatch(
                vectors=((0.1, 0.2, 0.3), (0.1, 0.2)),
                model="m",
                dimensions=3,
                total_tokens=4,
            )


def test_token_usage_total() -> None:
    assert TokenUsage(input_tokens=100, output_tokens=50).total_tokens == 150


def test_refusal_is_a_value_not_an_error() -> None:
    refusal = Refusal(reason="below confidence threshold", best_score=0.12)
    assert refusal.best_score == 0.12
