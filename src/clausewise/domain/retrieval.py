"""Retrieval-side domain values: vectors, embedding batches, scored results."""

from dataclasses import dataclass
from enum import StrEnum

from clausewise.domain.documents import Chunk

# A dense embedding. Tuple (not list) keeps domain objects hashable/immutable.
Vector = tuple[float, ...]


class RetrievalSource(StrEnum):
    """Which stage produced a score. Scores are NOT comparable across sources."""

    DENSE = "dense"
    KEYWORD = "keyword"
    FUSED = "fused"
    RERANKED = "reranked"


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Result of embedding a batch of texts, with cost accounting built in."""

    vectors: tuple[Vector, ...]
    model: str
    dimensions: int
    total_tokens: int

    def __post_init__(self) -> None:
        bad = [i for i, v in enumerate(self.vectors) if len(v) != self.dimensions]
        if bad:
            msg = f"Vectors at indices {bad} do not match dimensions={self.dimensions}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk plus the score and provenance of how it was retrieved."""

    chunk: Chunk
    score: float
    source: RetrievalSource
