"""Metadata enrichment: attach CUAD clause-type labels to overlapping chunks.

A chunk gets a clause-type label when its source span overlaps an expert-
annotated span for that type. Labels are stored for analysis and optional
filtering — they are NEVER queryable during evaluation runs (label leakage
would invalidate every eval number; see PROJECT_PLAN Ckpt 7/15).
"""

from dataclasses import replace

from clausewise.domain import Chunk
from clausewise.ingestion.cuad import CuadContract


def enrich_with_clause_types(chunks: list[Chunk], cuad: CuadContract) -> list[Chunk]:
    """Return chunks with ``metadata.clause_types`` filled from annotations."""
    spans = [
        (span.char_start, span.char_end, ann.clause_type)
        for ann in cuad.annotations
        for span in ann.spans
    ]
    if not spans:
        return chunks

    enriched: list[Chunk] = []
    for chunk in chunks:
        labels = sorted({ct for start, end, ct in spans if chunk.overlaps(start, end)})
        if labels:
            enriched.append(
                replace(chunk, metadata=replace(chunk.metadata, clause_types=tuple(labels)))
            )
        else:
            enriched.append(chunk)
    return enriched
