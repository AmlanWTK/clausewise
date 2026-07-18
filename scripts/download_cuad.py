"""Download CUAD and convert to our canonical JSONL layout.

Fetches ``data.zip`` from the official CUAD GitHub repository (SQuAD-format
JSON) and normalizes it — no third-party dependencies required.

Usage:
    uv run python scripts/download_cuad.py [--out data/cuad]

Produces:
    data/cuad/contracts.jsonl    one line per contract: {title, text}
    data/cuad/annotations.jsonl  one line per (contract, clause type)
    data/cuad/manifest.json      counts + content fingerprint for reproducibility

CUAD v1 (c) The Atticus Project, Inc., CC-BY 4.0.
"""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clausewise.ingestion.cuad import clause_type_from_row_id

DATA_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def download_zip(url: str) -> bytes:
    print(f"Downloading {url} ...")
    if not url.startswith("https://"):  # defense-in-depth for S310
        raise SystemExit("Refusing non-https URL")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - https enforced above
        payload: bytes = response.read()
    print(f"Downloaded {len(payload) / 1e6:.1f} MB")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cuad"))
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    payload = download_zip(DATA_ZIP_URL)

    texts: dict[str, str] = {}
    # (title, clause_type) -> {"question": str, "spans": {(start, text), ...}}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # data.zip contains CUADv1.json (the complete, authoritative dataset)
        # PLUS train/test split files that re-encode the same QAs with suffixed
        # ids. Parsing all of them double-counts and pollutes clause-type
        # labels — use only CUADv1.json.
        all_json = [n for n in archive.namelist() if n.endswith(".json")]
        json_names = [n for n in all_json if n.split("/")[-1] == "CUADv1.json"]
        if not json_names:
            raise SystemExit(f"FATAL: CUADv1.json not found in data.zip (saw: {all_json})")
        print(f"Parsing: {', '.join(json_names)}")
        for name in json_names:
            squad = json.loads(archive.read(name))
            for entry in squad["data"]:
                title = entry["title"]
                for paragraph in entry["paragraphs"]:
                    context = paragraph["context"]
                    if title in texts and texts[title] != context:
                        raise SystemExit(
                            f"FATAL: contract {title!r} has differing context across files — "
                            "canonical-text invariant would break."
                        )
                    texts[title] = context
                    for qa in paragraph["qas"]:
                        row_count += 1
                        key = (title, clause_type_from_row_id(qa["id"]))
                        record = merged.setdefault(
                            key, {"question": qa["question"], "spans": set()}
                        )
                        for answer in qa["answers"]:
                            record["spans"].add((answer["answer_start"], answer["text"]))

    with (out / "contracts.jsonl").open("w", encoding="utf-8") as f:
        for title in sorted(texts):
            f.write(json.dumps({"title": title, "text": texts[title]}) + "\n")

    with (out / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for (title, clause_type), record in sorted(merged.items()):
            spans = [{"text": text, "char_start": start} for start, text in sorted(record["spans"])]
            f.write(
                json.dumps(
                    {
                        "title": title,
                        "clause_type": clause_type,
                        "question": record["question"],
                        "spans": spans,
                    }
                )
                + "\n"
            )

    fingerprint = hashlib.sha256(
        "".join(f"{t}:{len(texts[t])}" for t in sorted(texts)).encode()
    ).hexdigest()
    manifest = {
        "source": DATA_ZIP_URL,
        "license": "CC-BY 4.0 (The Atticus Project, Inc.)",
        "contracts": len(texts),
        "annotation_rows": len(merged),
        "source_qa_rows": row_count,
        "fingerprint": fingerprint,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done: {len(texts)} contracts, {len(merged)} annotation rows -> {out}")
    print(f"Fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
