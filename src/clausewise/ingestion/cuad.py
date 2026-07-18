"""CUAD dataset models and loader.

The download script (scripts/download_cuad.py) converts the Hugging Face
``theatticusproject/cuad-qa`` dataset into two JSONL files under
``data/cuad/``:

- ``contracts.jsonl``   — one line per contract: {"title", "text"}
- ``annotations.jsonl`` — one line per (contract, clause type):
    {"title", "clause_type", "question", "spans": [{"text", "char_start"}]}

CRITICAL INVARIANT — canonical text: annotation offsets index into the exact
``text`` string stored in contracts.jsonl (the HF ``context`` field). The
loader verifies every span against the text and fails loudly on mismatch;
downstream parsing/chunking must never normalize this string in place.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from clausewise.domain import Contract
from clausewise.domain.errors import IngestionError

CUAD_DIR_DEFAULT = Path("data/cuad")


@dataclass(frozen=True, slots=True)
class CuadSpan:
    """One expert-annotated span, offsets into the contract's canonical text."""

    text: str
    char_start: int

    @property
    def char_end(self) -> int:
        return self.char_start + len(self.text)


@dataclass(frozen=True, slots=True)
class CuadAnnotation:
    """All annotated spans for one clause type in one contract.

    ``spans`` may be empty — CUAD explicitly marks clause types that are
    absent from a contract. These "no-answer" cases become the refusal
    portion of the evaluation set (PROJECT_PLAN Ckpt 15).
    """

    clause_type: str
    question: str
    spans: tuple[CuadSpan, ...]

    @property
    def has_answer(self) -> bool:
        return len(self.spans) > 0


@dataclass(frozen=True, slots=True)
class CuadContract:
    """A contract with its canonical text and expert annotations."""

    title: str
    text: str
    annotations: tuple[CuadAnnotation, ...]


def contract_id_for(text: str) -> str:
    """Stable content-derived contract id (sha256 prefix of canonical text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def to_domain_contract(cuad: CuadContract) -> Contract:
    """Convert a CUAD record to the domain Contract (annotations stay behind)."""
    return Contract(id=contract_id_for(cuad.text), title=cuad.title, text=cuad.text)


def clause_type_from_row_id(row_id: str) -> str:
    """Extract the clause-type label from a HF row id.

    Row ids look like ``"<CONTRACT NAME>__<Clause Type>"``; the suffix after
    the double underscore is the label (e.g. "Governing Law").
    """
    _, sep, clause_type = row_id.rpartition("__")
    if not sep or not clause_type:
        msg = f"Cannot extract clause type from row id: {row_id!r}"
        raise IngestionError(msg)
    return clause_type


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        msg = f"Missing {path} — run: uv run python scripts/download_cuad.py"
        raise IngestionError(msg)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"{path}:{line_no}: invalid JSON"
                    raise IngestionError(msg) from exc


def load_cuad(data_dir: Path = CUAD_DIR_DEFAULT) -> list[CuadContract]:
    """Load contracts + annotations from JSONL, verifying span integrity.

    Every span's text must equal the exact slice of the canonical contract
    text it claims to occupy. A mismatch means the canonical-text invariant
    was broken somewhere — that corrupts all evaluation numbers, so we fail
    immediately rather than continue with bad ground truth.
    """
    texts: dict[str, str] = {}
    for row in _read_jsonl(data_dir / "contracts.jsonl"):
        texts[str(row["title"])] = str(row["text"])

    annotations: dict[str, list[CuadAnnotation]] = {title: [] for title in texts}
    for row in _read_jsonl(data_dir / "annotations.jsonl"):
        title = str(row["title"])
        if title not in texts:
            msg = f"Annotation references unknown contract: {title!r}"
            raise IngestionError(msg)
        raw_spans = row.get("spans", [])
        if not isinstance(raw_spans, list):
            msg = f"Malformed spans for {title!r}"
            raise IngestionError(msg)
        spans = tuple(
            CuadSpan(text=str(s["text"]), char_start=int(s["char_start"])) for s in raw_spans
        )
        for span in spans:
            actual = texts[title][span.char_start : span.char_end]
            if actual != span.text:
                msg = (
                    f"Span integrity violation in {title!r} at {span.char_start}: "
                    f"expected {span.text[:60]!r}, found {actual[:60]!r}"
                )
                raise IngestionError(msg)
        annotations[title].append(
            CuadAnnotation(
                clause_type=str(row["clause_type"]),
                question=str(row["question"]),
                spans=spans,
            )
        )

    return [
        CuadContract(title=title, text=text, annotations=tuple(annotations[title]))
        for title, text in sorted(texts.items())
    ]
