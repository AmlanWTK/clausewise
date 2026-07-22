"""Shared chunking utilities: config, token counting, ids, sentence splits."""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

# Counts tokens in a text. Injectable so the chunker can use the *actual*
# embedding-model tokenizer (wired in at Checkpoint 8) instead of a guess.
TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Cheap default token estimate (~4 chars/token for English legal text).

    Deliberately conservative; the real tokenizer replaces this at embedding
    time. Only chunk-size decisions depend on it, and configs leave headroom
    against the embedding model's hard 512-token window.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Size policy for chunkers. Ablation variants are different configs."""

    target_tokens: int = 350  # aim here when splitting long sections
    min_tokens: int = 40  # merge units smaller than this into a neighbor
    max_tokens: int = 480  # hard cap (headroom under BGE's 512 window)
    overlap_tokens: int = 60  # fixed-size baseline only

    def __post_init__(self) -> None:
        if not (0 < self.min_tokens <= self.target_tokens <= self.max_tokens):
            msg = f"Inconsistent chunker config: {self}"
            raise ValueError(msg)


def chunk_id(contract_id: str, strategy: str, char_start: int, char_end: int) -> str:
    """Deterministic chunk id — same span + strategy always hashes the same,
    which is what makes ingestion re-runs idempotent."""
    raw = f"{contract_id}:{strategy}:{char_start}:{char_end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+")


def sentence_spans(text: str, base_offset: int = 0) -> list[tuple[int, int]]:
    """Split text into sentence-ish spans, offsets relative to the original.

    Splits after . ; : followed by whitespace — crude but adequate for legal
    prose, and never breaks mid-word.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        if m.start() > start:
            spans.append((base_offset + start, base_offset + m.start()))
        start = m.end()
    if start < len(text):
        spans.append((base_offset + start, base_offset + len(text)))
    return spans
