"""Unit tests for Reciprocal Rank Fusion — hand-computed fixtures."""

import pytest

from clausewise.domain import Chunk, ChunkMetadata, RetrievedChunk
from clausewise.domain.retrieval import RetrievalSource
from clausewise.retrieval import reciprocal_rank_fusion


def _result(chunk_id: str, score: float, source: RetrievalSource) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            contract_id="k1",
            text=f"text {chunk_id}",
            char_start=0,
            char_end=10,
            token_count=2,
            metadata=ChunkMetadata(contract_title="T"),
        ),
        score=score,
        source=source,
    )


def _dense(*ids: str) -> list[RetrievedChunk]:
    return [_result(i, 1.0 - n * 0.1, RetrievalSource.DENSE) for n, i in enumerate(ids)]


def _keyword(*ids: str) -> list[RetrievedChunk]:
    return [_result(i, 9.0 - n, RetrievalSource.KEYWORD) for n, i in enumerate(ids)]


def test_hand_computed_scores() -> None:
    # a: rank 1 in both → 2/61. b: rank 2 dense only → 1/62. c: rank 2 kw only → 1/62.
    fused = reciprocal_rank_fusion([_dense("a", "b"), _keyword("a", "c")], k=10)
    by_id = {r.chunk.id: r.score for r in fused}
    assert by_id["a"] == pytest.approx(2 / 61)
    assert by_id["b"] == pytest.approx(1 / 62)
    assert by_id["c"] == pytest.approx(1 / 62)
    assert fused[0].chunk.id == "a"


def test_agreement_beats_single_list_top_rank() -> None:
    # d is #1 in dense only; x is #2 in BOTH lists — x must win:
    # x: 2/62 ≈ 0.0323 > d: 1/61 ≈ 0.0164
    fused = reciprocal_rank_fusion([_dense("d", "x"), _keyword("y", "x")], k=10)
    assert fused[0].chunk.id == "x"


def test_weights_shift_the_balance() -> None:
    lists = [_dense("a"), _keyword("b")]
    fused = reciprocal_rank_fusion(lists, k=2, weights=[3.0, 1.0])
    assert fused[0].chunk.id == "a"
    fused = reciprocal_rank_fusion(lists, k=2, weights=[1.0, 3.0])
    assert fused[0].chunk.id == "b"


def test_output_is_fused_source_and_truncated() -> None:
    fused = reciprocal_rank_fusion([_dense("a", "b", "c"), _keyword("c", "d")], k=2)
    assert len(fused) == 2
    assert all(r.source is RetrievalSource.FUSED for r in fused)


def test_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []], k=5) == []


def test_mismatched_weights_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([_dense("a")], weights=[1.0, 2.0])
