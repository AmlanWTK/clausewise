"""Chunker port."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from clausewise.domain import Chunk, Contract, Section


@runtime_checkable
class Chunker(Protocol):
    """Splits a contract into retrievable chunks.

    ``sections`` is the structural tree from the parser (Checkpoint 5); the
    fixed-size baseline chunker ignores it, the clause-aware chunker follows it.
    Implementations must preserve char-offset lineage into Contract.text.
    """

    @property
    def name(self) -> str:
        """Strategy identifier, used as the corpus name (e.g. 'clause_aware')."""
        ...

    def chunk(self, contract: Contract, sections: Sequence[Section]) -> list[Chunk]: ...
