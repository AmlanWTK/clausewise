"""Fixed-size token-window chunker — the ablation baseline.

Deliberately naive: slides a token window across the raw text with overlap,
ignoring all document structure. This is what most RAG tutorials do; the
ablation study (Checkpoint 16) measures exactly what that naivety costs.
"""

import re
from collections.abc import Sequence

from clausewise.domain import Chunk, ChunkMetadata, Contract, Section
from clausewise.ingestion.chunkers.common import (
    ChunkerConfig,
    TokenCounter,
    chunk_id,
    estimate_tokens,
)

_WORD_RE = re.compile(r"\S+")


class FixedSizeChunker:
    """Token windows with overlap; offsets preserved, structure ignored."""

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        token_counter: TokenCounter = estimate_tokens,
    ) -> None:
        self._config = config or ChunkerConfig()
        self._count = token_counter

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(self, contract: Contract, sections: Sequence[Section]) -> list[Chunk]:
        del sections  # baseline ignores structure by design
        text = contract.text
        words = list(_WORD_RE.finditer(text))
        if not words:
            return []

        target = self._config.target_tokens
        overlap = self._config.overlap_tokens
        chunks: list[Chunk] = []
        start_idx = 0

        while start_idx < len(words):
            token_total = 0
            end_idx = start_idx
            while end_idx < len(words) and token_total < target:
                token_total += self._count(words[end_idx].group())
                end_idx += 1

            char_start = words[start_idx].start()
            char_end = words[end_idx - 1].end()
            piece = text[char_start:char_end]
            chunks.append(
                Chunk(
                    id=chunk_id(contract.id, self.name, char_start, char_end),
                    contract_id=contract.id,
                    text=piece,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=self._count(piece),
                    metadata=ChunkMetadata(contract_title=contract.title, parties=contract.parties),
                )
            )

            if end_idx >= len(words):
                break
            # Step back ~overlap tokens worth of words for the next window.
            back = 0
            back_tokens = 0
            while back < end_idx - start_idx - 1 and back_tokens < overlap:
                back += 1
                back_tokens += self._count(words[end_idx - back].group())
            start_idx = end_idx - back

        return chunks
