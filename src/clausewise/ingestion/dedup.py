"""Near-duplicate chunk detection via SimHash.

Boilerplate clauses repeat near-verbatim across contracts (notice provisions,
severability, counterparts...). Duplicates are *marked*, never dropped — every
chunk keeps its provenance, and retrieval can collapse duplicates at query
time without losing the fact that contract X contains the clause.

Implementation: 64-bit SimHash over 5-gram character shingles of normalized
text. Candidate pairs come from LSH banding (4 bands x 16 bits); a pair is a
duplicate if Hamming distance <= threshold. Zero dependencies, O(n) buckets.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence

from clausewise.domain import Chunk

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")

_SHINGLE = 5
_BANDS = 4
_BAND_BITS = 16
_HAMMING_THRESHOLD = 3


def normalize(text: str) -> str:
    """Normalize for similarity: lowercase, collapse whitespace, mask numbers.

    Masking digits makes "within 30 days" and "within 60 days" compare equal —
    intentional: boilerplate with different constants is still boilerplate.
    """
    text = text.lower()
    text = _NUM_RE.sub("0", text)
    return _WS_RE.sub(" ", text).strip()


def simhash64(text: str) -> int:
    """64-bit SimHash over character shingles."""
    weights = [0] * 64
    normalized = normalize(text)
    if len(normalized) < _SHINGLE:
        shingles: list[str] = [normalized]
    else:
        shingles = [normalized[i : i + _SHINGLE] for i in range(len(normalized) - _SHINGLE + 1)]
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit in range(64):
        if weights[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_duplicates(chunks: Sequence[Chunk]) -> dict[str, str]:
    """Map duplicate chunk id -> canonical chunk id.

    Canonical = first chunk seen (stable input order). Chunks from the same
    contract are never marked as duplicates of each other — internal
    repetition is real structure, not cross-contract boilerplate.
    """
    hashes: dict[str, int] = {c.id: simhash64(c.text) for c in chunks}
    buckets: dict[tuple[int, int], list[Chunk]] = defaultdict(list)
    duplicate_of: dict[str, str] = {}

    for chunk in chunks:
        if chunk.id in duplicate_of:
            continue
        h = hashes[chunk.id]
        assigned = False
        candidate_keys = [
            (band, (h >> (band * _BAND_BITS)) & ((1 << _BAND_BITS) - 1)) for band in range(_BANDS)
        ]
        for key in candidate_keys:
            for canonical in buckets[key]:
                if canonical.contract_id == chunk.contract_id:
                    continue
                if canonical.id in duplicate_of:
                    continue
                if hamming(h, hashes[canonical.id]) <= _HAMMING_THRESHOLD:
                    duplicate_of[chunk.id] = canonical.id
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            for key in candidate_keys:
                buckets[key].append(chunk)

    return duplicate_of
