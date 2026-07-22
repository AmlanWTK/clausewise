"""Unit tests for SimHash near-duplicate detection."""

from clausewise.domain import Chunk, ChunkMetadata
from clausewise.ingestion.dedup import (
    HAMMING_THRESHOLD,
    find_duplicates,
    hamming,
    normalize,
    simhash64,
)

BOILERPLATE = (
    "This Agreement may be executed in counterparts, each of which shall be "
    "deemed an original, but all of which together shall constitute one and "
    "the same instrument. Any notice required hereunder shall be in writing "
    "and delivered by certified mail to the addresses set forth above."
)


def _chunk(chunk_id: str, text: str, contract_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        contract_id=contract_id,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text) // 4,
        metadata=ChunkMetadata(contract_title="T"),
    )


def test_normalize_masks_numbers_and_whitespace() -> None:
    assert normalize("Within  30\ndays") == normalize("within 60 DAYS")


def test_identical_text_same_hash() -> None:
    assert simhash64(BOILERPLATE) == simhash64(BOILERPLATE)


def test_near_identical_text_close_hash() -> None:
    variant = BOILERPLATE.replace("certified mail", "registered mail")
    assert hamming(simhash64(BOILERPLATE), simhash64(variant)) <= HAMMING_THRESHOLD


def test_different_text_distant_hash() -> None:
    other = (
        "Licensor grants Licensee an exclusive, non-transferable license to use "
        "the Software solely for internal business purposes in the Territory."
    )
    assert hamming(simhash64(BOILERPLATE), simhash64(other)) > 10


def test_cross_contract_duplicates_marked() -> None:
    chunks = [
        _chunk("c1", BOILERPLATE, "contract_a"),
        _chunk("c2", "Completely unrelated payment obligations text here.", "contract_a"),
        _chunk("c3", BOILERPLATE, "contract_b"),
        _chunk("c4", BOILERPLATE.replace("certified", "registered"), "contract_c"),
    ]
    duplicate_of = find_duplicates(chunks)
    assert duplicate_of.get("c3") == "c1"
    assert duplicate_of.get("c4") == "c1"
    assert "c1" not in duplicate_of  # canonical stays canonical
    assert "c2" not in duplicate_of


def test_same_contract_repetition_is_not_deduped() -> None:
    chunks = [
        _chunk("c1", BOILERPLATE, "contract_a"),
        _chunk("c2", BOILERPLATE, "contract_a"),  # same contract — keep both
    ]
    assert find_duplicates(chunks) == {}


def test_empty_and_tiny_inputs() -> None:
    assert find_duplicates([]) == {}
    tiny = [_chunk("c1", "ok", "a"), _chunk("c2", "ok", "b")]
    assert find_duplicates(tiny).get("c2") == "c1"
