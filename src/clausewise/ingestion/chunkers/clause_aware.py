"""Clause-aware chunker — follows the parsed legal structure.

Strategy:
1. Walk the section tree; each node's "own text" (its span minus its
   children's spans) becomes a candidate unit, tagged with its section path.
   Text outside any section (preamble, recitals) becomes its own unit(s).
2. Units under ``min_tokens`` merge into the previous unit of the same
   section path's parent scope — a 6-word heading is retrieval noise alone.
3. Units over ``max_tokens`` split at sentence boundaries into pieces
   near ``target_tokens``.
4. Every chunk's text is prefixed with its section path ("[ARTICLE II > 2.1]")
   so retrieval sees context, while char offsets always point at the raw
   source span in the canonical text (the prefix is display/embedding only).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from clausewise.domain import Chunk, ChunkMetadata, Contract, Section
from clausewise.ingestion.chunkers.common import (
    ChunkerConfig,
    TokenCounter,
    chunk_id,
    estimate_tokens,
    sentence_spans,
)


@dataclass(slots=True)
class _Unit:
    """A candidate chunk: a source span plus its section path."""

    char_start: int
    char_end: int
    path: tuple[str, ...]


class ClauseAwareChunker:
    """Chunks along legal structure with merge/split size normalization."""

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        token_counter: TokenCounter = estimate_tokens,
    ) -> None:
        self._config = config or ChunkerConfig()
        self._count = token_counter

    @property
    def name(self) -> str:
        return "clause_aware"

    def chunk(self, contract: Contract, sections: Sequence[Section]) -> list[Chunk]:
        text = contract.text
        units = self._collect_units(text, sections)
        units = self._merge_small(text, units)
        units = self._split_large(text, units)

        chunks: list[Chunk] = []
        for unit in units:
            source = text[unit.char_start : unit.char_end]
            if not source.strip():
                continue
            prefix = f"[{' > '.join(unit.path)}] " if unit.path else ""
            chunks.append(
                Chunk(
                    id=chunk_id(contract.id, self.name, unit.char_start, unit.char_end),
                    contract_id=contract.id,
                    text=prefix + source,
                    char_start=unit.char_start,
                    char_end=unit.char_end,
                    token_count=self._count(prefix + source),
                    metadata=ChunkMetadata(
                        contract_title=contract.title,
                        parties=contract.parties,
                        section_path=unit.path,
                    ),
                )
            )
        return chunks

    # --- stage 1: structure walk -------------------------------------------

    def _collect_units(self, text: str, sections: Sequence[Section]) -> list[_Unit]:
        units: list[_Unit] = []

        def walk(section: Section, path: tuple[str, ...]) -> None:
            here = (*path, section.number)
            cursor = section.char_start
            for child in section.children:
                if child.char_start > cursor:
                    units.append(_Unit(cursor, child.char_start, here))
                walk(child, here)
                cursor = max(cursor, child.char_end)
            if cursor < section.char_end:
                units.append(_Unit(cursor, section.char_end, here))

        cursor = 0
        for top in sections:
            if top.char_start > cursor:
                units.append(_Unit(cursor, top.char_start, ()))  # preamble/gap
            walk(top, ())
            cursor = max(cursor, top.char_end)
        if cursor < len(text):
            units.append(_Unit(cursor, len(text), ()))
        return [u for u in units if text[u.char_start : u.char_end].strip()]

    # --- stage 2: merge tiny units -----------------------------------------

    def _merge_small(self, text: str, units: list[_Unit]) -> list[_Unit]:
        merged: list[_Unit] = []
        i = 0
        while i < len(units):
            unit = units[i]
            tokens = self._count(text[unit.char_start : unit.char_end])
            if tokens >= self._config.min_tokens:
                merged.append(unit)
                i += 1
                continue

            nxt = units[i + 1] if i + 1 < len(units) else None
            if (
                nxt is not None
                and unit.char_end == nxt.char_start
                and unit.path == nxt.path[: len(unit.path)]
            ):
                # Small parent/heading unit ("ARTICLE II - OBLIGATIONS") —
                # merge FORWARD into its first child, keeping the child's
                # path. Merging backward would glue a heading onto the
                # previous article's last clause.
                units[i + 1] = _Unit(unit.char_start, nxt.char_end, nxt.path)
                i += 1
                continue

            if merged and merged[-1].char_end == unit.char_start:
                # Trailing fragment — absorb into the previous adjacent unit.
                prev = merged[-1]
                keep = prev.path if len(prev.path) <= len(unit.path) else unit.path
                merged[-1] = _Unit(prev.char_start, unit.char_end, keep)
                i += 1
                continue

            merged.append(unit)  # small but isolated: keep as-is
            i += 1
        return merged

    # --- stage 3: split oversized units ------------------------------------

    def _split_large(self, text: str, units: list[_Unit]) -> list[_Unit]:
        out: list[_Unit] = []
        for unit in units:
            source = text[unit.char_start : unit.char_end]
            if self._count(source) <= self._config.max_tokens:
                out.append(unit)
                continue
            piece_start = unit.char_start
            piece_tokens = 0
            for s_start, s_end in sentence_spans(source, unit.char_start):
                sentence_tokens = self._count(text[s_start:s_end])
                if piece_tokens + sentence_tokens > self._config.target_tokens and (
                    piece_start < s_start
                ):
                    out.append(_Unit(piece_start, s_start, unit.path))
                    piece_start = s_start
                    piece_tokens = 0
                piece_tokens += sentence_tokens
            if piece_start < unit.char_end:
                out.append(_Unit(piece_start, unit.char_end, unit.path))
        return out
