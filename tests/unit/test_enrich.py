"""Unit tests for clause-type enrichment."""

from clausewise.domain import Chunk, ChunkMetadata
from clausewise.ingestion.cuad import CuadAnnotation, CuadContract, CuadSpan
from clausewise.ingestion.enrich import enrich_with_clause_types


def _chunk(chunk_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        contract_id="k1",
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        token_count=5,
        metadata=ChunkMetadata(contract_title="T"),
    )


def _cuad(spans: dict[str, tuple[int, int]]) -> CuadContract:
    text = "y" * 1000
    return CuadContract(
        title="T",
        text=text,
        annotations=tuple(
            CuadAnnotation(
                clause_type=ct,
                question=f"about {ct}",
                spans=(CuadSpan(text=text[s:e], char_start=s),),
            )
            for ct, (s, e) in spans.items()
        ),
    )


def test_overlapping_chunk_gets_label() -> None:
    chunks = [_chunk("c1", 0, 100), _chunk("c2", 100, 200)]
    cuad = _cuad({"Governing Law": (50, 80)})
    enriched = enrich_with_clause_types(chunks, cuad)
    assert enriched[0].metadata.clause_types == ("Governing Law",)
    assert enriched[1].metadata.clause_types == ()


def test_span_straddling_two_chunks_labels_both() -> None:
    chunks = [_chunk("c1", 0, 100), _chunk("c2", 100, 200)]
    cuad = _cuad({"Anti-Assignment": (90, 110)})
    enriched = enrich_with_clause_types(chunks, cuad)
    assert enriched[0].metadata.clause_types == ("Anti-Assignment",)
    assert enriched[1].metadata.clause_types == ("Anti-Assignment",)


def test_multiple_labels_sorted() -> None:
    chunks = [_chunk("c1", 0, 300)]
    cuad = _cuad({"Governing Law": (10, 20), "Audit Rights": (50, 60)})
    enriched = enrich_with_clause_types(chunks, cuad)
    assert enriched[0].metadata.clause_types == ("Audit Rights", "Governing Law")


def test_no_annotations_is_a_noop() -> None:
    chunks = [_chunk("c1", 0, 100)]
    cuad = CuadContract(title="T", text="y" * 200, annotations=())
    assert enrich_with_clause_types(chunks, cuad) == chunks
