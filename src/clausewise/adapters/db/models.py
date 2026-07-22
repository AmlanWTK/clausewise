"""SQLAlchemy ORM models.

Rows are persistence records, not domain objects — converters live next to
the models. The embedding column is added by a later migration (Checkpoint 8);
these tables carry everything ingestion produces.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from clausewise.domain import Chunk, ChunkMetadata, Contract


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ContractRow(Base):
    __tablename__ = "contracts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    parties: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source: Mapped[str] = mapped_column(Text, default="CUAD", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    @staticmethod
    def from_domain(contract: Contract) -> "ContractRow":
        return ContractRow(
            id=contract.id,
            title=contract.title,
            text=contract.text,
            parties=list(contract.parties),
            source=contract.source,
        )

    def to_domain(self) -> Contract:
        return Contract(
            id=self.id,
            title=self.title,
            text=self.text,
            parties=tuple(self.parties),
            source=self.source,
        )


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_corpus", "corpus"),
        Index("ix_chunks_contract", "contract_id"),
        Index("ix_chunks_corpus_contract", "corpus", "contract_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    corpus: Mapped[str] = mapped_column(Text, nullable=False)  # chunker strategy
    contract_id: Mapped[str] = mapped_column(
        ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    clause_types: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    duplicate_of: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    @staticmethod
    def from_domain(chunk: Chunk, corpus: str, contract_title: str) -> "ChunkRow":
        del contract_title  # title travels in metadata already
        return ChunkRow(
            id=chunk.id,
            corpus=corpus,
            contract_id=chunk.contract_id,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            token_count=chunk.token_count,
            section_path=list(chunk.metadata.section_path),
            clause_types=list(chunk.metadata.clause_types),
            duplicate_of=chunk.metadata.duplicate_of,
        )

    def to_domain(self, contract_title: str, parties: tuple[str, ...] = ()) -> Chunk:
        return Chunk(
            id=self.id,
            contract_id=self.contract_id,
            text=self.text,
            char_start=self.char_start,
            char_end=self.char_end,
            token_count=self.token_count,
            metadata=ChunkMetadata(
                contract_title=contract_title,
                parties=parties,
                section_path=tuple(self.section_path),
                clause_types=tuple(self.clause_types),
                duplicate_of=self.duplicate_of,
            ),
        )
