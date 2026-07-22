"""Chunkers: implementations of the Chunker port.

- ClauseAwareChunker — follows the parsed section tree (the differentiator).
- FixedSizeChunker — token-window baseline, kept for the ablation study.
"""

from clausewise.ingestion.chunkers.clause_aware import ClauseAwareChunker
from clausewise.ingestion.chunkers.common import ChunkerConfig, TokenCounter, estimate_tokens
from clausewise.ingestion.chunkers.fixed_size import FixedSizeChunker

__all__ = [
    "ChunkerConfig",
    "ClauseAwareChunker",
    "FixedSizeChunker",
    "TokenCounter",
    "estimate_tokens",
]
