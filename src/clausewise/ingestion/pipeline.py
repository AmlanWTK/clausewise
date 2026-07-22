"""Staged ingestion pipeline: CUAD → parse → chunk → enrich → dedup → persist.

Idempotency: chunk ids are content-derived, and persistence uses
INSERT ... ON CONFLICT DO NOTHING — re-running the pipeline is always safe
and a completed run is a no-op. Per-stage metrics are logged and returned.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from clausewise.adapters.db import ChunkRow, ContractRow
from clausewise.adapters.db.engine import session_scope
from clausewise.domain import Chunk
from clausewise.ingestion.chunkers import ClauseAwareChunker, FixedSizeChunker
from clausewise.ingestion.cuad import (
    CUAD_DIR_DEFAULT,
    CuadContract,
    load_cuad,
    to_domain_contract,
)
from clausewise.ingestion.dedup import find_duplicates
from clausewise.ingestion.enrich import enrich_with_clause_types
from clausewise.ingestion.parser import parse_contract
from clausewise.observability import get_logger

if TYPE_CHECKING:
    from clausewise.ports import Chunker

log = get_logger(__name__)

CHUNKERS: dict[str, type[ClauseAwareChunker] | type[FixedSizeChunker]] = {
    "clause_aware": ClauseAwareChunker,
    "fixed_size": FixedSizeChunker,
}


@dataclass(slots=True)
class IngestStats:
    contracts: int = 0
    chunks: int = 0
    duplicates: int = 0
    enriched: int = 0
    written_contracts: int = 0
    written_chunks: int = 0
    per_corpus: dict[str, int] = field(default_factory=dict)


def run_ingestion(
    session_factory: sessionmaker[Session],
    *,
    corpora: list[str],
    data_dir: Path = CUAD_DIR_DEFAULT,
    limit: int | None = None,
    dry_run: bool = False,
) -> IngestStats:
    """Run the full pipeline for the requested chunker strategies."""
    unknown = set(corpora) - CHUNKERS.keys()
    if unknown:
        msg = f"Unknown corpora: {sorted(unknown)}; valid: {sorted(CHUNKERS)}"
        raise ValueError(msg)

    stats = IngestStats()
    cuad_contracts = load_cuad(data_dir)
    if limit is not None:
        cuad_contracts = cuad_contracts[:limit]
    stats.contracts = len(cuad_contracts)
    log.info("ingest_start", contracts=stats.contracts, corpora=corpora, dry_run=dry_run)

    for corpus in corpora:
        chunker: Chunker = CHUNKERS[corpus]()
        all_chunks: list[Chunk] = []

        for cuad in cuad_contracts:
            contract = to_domain_contract(cuad)
            sections = parse_contract(contract)
            chunks = chunker.chunk(contract, sections)
            chunks = enrich_with_clause_types(chunks, cuad)
            all_chunks.extend(chunks)

        stats.enriched += sum(1 for c in all_chunks if c.metadata.clause_types)

        duplicate_of = find_duplicates(all_chunks)
        stats.duplicates += len(duplicate_of)
        all_chunks = [
            replace(c, metadata=replace(c.metadata, duplicate_of=duplicate_of.get(c.id)))
            if c.id in duplicate_of
            else c
            for c in all_chunks
        ]

        stats.chunks += len(all_chunks)
        stats.per_corpus[corpus] = len(all_chunks)
        log.info(
            "corpus_chunked",
            corpus=corpus,
            chunks=len(all_chunks),
            duplicates=len(duplicate_of),
        )

        if dry_run:
            continue

        with session_scope(session_factory) as session:
            stats.written_contracts = _persist_contracts(session, cuad_contracts)
            stats.written_chunks += _persist_chunks(session, all_chunks, corpus)

    log.info(
        "ingest_done",
        chunks=stats.chunks,
        duplicates=stats.duplicates,
        written_chunks=stats.written_chunks,
    )
    return stats


_BATCH = 500


def _persist_contracts(session: Session, cuad_contracts: list[CuadContract]) -> int:
    written = 0
    rows = [ContractRow.from_domain(to_domain_contract(c)) for c in cuad_contracts]
    for i in range(0, len(rows), _BATCH):
        batch = rows[i : i + _BATCH]
        stmt = (
            pg_insert(ContractRow)
            .values(
                [
                    {
                        "id": r.id,
                        "title": r.title,
                        "text": r.text,
                        "parties": r.parties,
                        "source": r.source,
                    }
                    for r in batch
                ]
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        written += session.execute(stmt).rowcount or 0
    return written


def _persist_chunks(session: Session, chunks: list[Chunk], corpus: str) -> int:
    written = 0
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i : i + _BATCH]
        stmt = (
            pg_insert(ChunkRow)
            .values(
                [
                    {
                        "id": c.id,
                        "corpus": corpus,
                        "contract_id": c.contract_id,
                        "text": c.text,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "token_count": c.token_count,
                        "section_path": list(c.metadata.section_path),
                        "clause_types": list(c.metadata.clause_types),
                        "duplicate_of": c.metadata.duplicate_of,
                    }
                    for c in batch
                ]
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        written += session.execute(stmt).rowcount or 0
    return written
