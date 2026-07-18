"""Document-side domain entities: contracts, sections, chunks.

Character offsets are always relative to the contract's canonical text
(``Contract.text``). Preserving this lineage end-to-end is what makes
evaluation against CUAD's span annotations possible (see PROJECT_PLAN,
Checkpoints 5 and 15).
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class Contract:
    """A single legal contract with its canonical text."""

    id: str  # stable content-derived identifier (e.g. sha256 prefix of text)
    title: str
    text: str  # canonical text; ALL char offsets refer to this exact string
    parties: tuple[str, ...] = ()
    source: str = "CUAD"
    filing_date: date | None = None


@dataclass(frozen=True, slots=True)
class Section:
    """A node in a contract's structural tree (article/section/sub-clause).

    ``number`` is the label as it appears in text ("2.1", "(a)", "ARTICLE IV").
    ``level`` is the depth in the hierarchy (0 = top-level article/section).
    Children are nested sections; offsets of children lie within the parent's.
    """

    number: str
    heading: str
    level: int
    char_start: int
    char_end: int
    children: tuple["Section", ...] = ()

    def __post_init__(self) -> None:
        if self.char_end < self.char_start:
            msg = (
                f"Section {self.number!r}: char_end {self.char_end} < char_start {self.char_start}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Structured metadata attached to every chunk.

    ``clause_types`` holds CUAD label names overlapping this chunk. They are
    stored for analysis/filtering but MUST NOT be used at query time when
    answering evaluation questions (label leakage — see PROJECT_PLAN Ckpt 7).
    """

    contract_title: str
    parties: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()  # e.g. ("ARTICLE IV", "4.2", "(a)")
    clause_types: tuple[str, ...] = ()
    duplicate_of: str | None = None  # chunk id of canonical copy, if deduplicated


@dataclass(frozen=True, slots=True)
class Chunk:
    """The retrievable unit: a span of contract text plus provenance."""

    id: str  # stable content-derived identifier
    contract_id: str
    text: str  # may include a context prefix (section heading)
    char_start: int  # offset of the *source span* in Contract.text
    char_end: int
    token_count: int
    metadata: ChunkMetadata = field(default=ChunkMetadata(contract_title=""))

    def __post_init__(self) -> None:
        if self.char_end < self.char_start:
            msg = f"Chunk {self.id!r}: char_end {self.char_end} < char_start {self.char_start}"
            raise ValueError(msg)
        if not self.text:
            msg = f"Chunk {self.id!r}: empty text"
            raise ValueError(msg)

    def overlaps(self, char_start: int, char_end: int) -> bool:
        """True if this chunk's source span overlaps [char_start, char_end).

        Used by the evaluation layer to map CUAD gold spans onto chunks.
        """
        return self.char_start < char_end and char_start < self.char_end
