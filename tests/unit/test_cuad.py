"""Unit tests for the CUAD loader — no network, fixture JSONL on disk."""

import json
from pathlib import Path

import pytest

from clausewise.domain.errors import IngestionError
from clausewise.ingestion.cuad import (
    clause_type_from_row_id,
    contract_id_for,
    load_cuad,
    to_domain_contract,
)

TEXT = "This Agreement shall be governed by the laws of Delaware."
# "the laws of Delaware" occupies TEXT[36:56].


def _write_fixture(
    tmp_path: Path,
    *,
    span_start: int = 36,
    span_text: str = "the laws of Delaware",
) -> Path:
    (tmp_path / "contracts.jsonl").write_text(
        json.dumps({"title": "ACME_MSA", "text": TEXT}) + "\n", encoding="utf-8"
    )
    (tmp_path / "annotations.jsonl").write_text(
        json.dumps(
            {
                "title": "ACME_MSA",
                "clause_type": "Governing Law",
                "question": 'Highlight the parts related to "Governing Law".',
                "spans": [{"text": span_text, "char_start": span_start}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "title": "ACME_MSA",
                "clause_type": "Anti-Assignment",
                "question": 'Highlight the parts related to "Anti-Assignment".',
                "spans": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_groups_annotations_by_contract(tmp_path: Path) -> None:
    contracts = load_cuad(_write_fixture(tmp_path))
    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.title == "ACME_MSA"
    assert len(contract.annotations) == 2
    governing = next(a for a in contract.annotations if a.clause_type == "Governing Law")
    assert governing.has_answer
    span = governing.spans[0]
    assert contract.text[span.char_start : span.char_end] == "the laws of Delaware"


def test_no_answer_annotation_is_preserved(tmp_path: Path) -> None:
    contracts = load_cuad(_write_fixture(tmp_path))
    anti_assignment = next(
        a for a in contracts[0].annotations if a.clause_type == "Anti-Assignment"
    )
    assert not anti_assignment.has_answer
    assert anti_assignment.spans == ()


def test_span_integrity_violation_fails_loudly(tmp_path: Path) -> None:
    # Off-by-two offset: the span text no longer matches the canonical slice.
    _write_fixture(tmp_path, span_start=35)
    with pytest.raises(IngestionError, match="Span integrity violation"):
        load_cuad(tmp_path)


def test_missing_files_give_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="download_cuad"):
        load_cuad(tmp_path)


def test_clause_type_from_row_id() -> None:
    row_id = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Governing Law"
    assert clause_type_from_row_id(row_id) == "Governing Law"


def test_clause_type_from_malformed_id_raises() -> None:
    with pytest.raises(IngestionError, match="Cannot extract clause type"):
        clause_type_from_row_id("no-separator-here")


def test_contract_id_is_content_derived(tmp_path: Path) -> None:
    contracts = load_cuad(_write_fixture(tmp_path))
    domain = to_domain_contract(contracts[0])
    assert domain.id == contract_id_for(TEXT)
    assert domain.text == TEXT  # canonical text passes through unmodified
