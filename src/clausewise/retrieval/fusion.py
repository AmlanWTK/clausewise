"""Reciprocal Rank Fusion (RRF).

Combines ranked lists using ranks only:

    score(d) = sum over lists i:  w_i / (rrf_k + rank_i(d))     (rank is 1-based)

Why ranks, not scores: dense cosine similarity and ts_rank_cd live on
incomparable scales; any score interpolation needs per-source calibration
that drifts with data. Ranks need none. rrf_k=60 is the standard constant
from Cormack et al. — it damps the head so one list's #1 can't steamroll
consistent mid-rank agreement from both lists.
"""

from collections.abc import Sequence
from dataclasses import replace

from clausewise.domain import RetrievedChunk
from clausewise.domain.retrieval import RetrievalSource

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into a single top-k list with source=FUSED."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        msg = f"weights ({len(weights)}) must match lists ({len(ranked_lists)})"
        raise ValueError(msg)

    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, item in enumerate(ranked, start=1):
            scores[item.chunk.id] = scores.get(item.chunk.id, 0.0) + weight / (rrf_k + rank)
            first_seen.setdefault(item.chunk.id, item)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        replace(first_seen[chunk_id], score=score, source=RetrievalSource.FUSED)
        for chunk_id, score in fused
    ]
