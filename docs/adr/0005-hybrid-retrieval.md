# ADR-0005: Hybrid retrieval — Postgres FTS + Reciprocal Rank Fusion

- **Status:** accepted
- **Date:** 2026-07-23

## Context

Legal Q&A needs exact lexical hits ("force majeure", "indemnification",
defined terms, party names) that dense embeddings blur into neighborhoods.
Pure vector search misses them; pure keyword search misses paraphrases
("who is responsible if things go wrong" → indemnification). We need both,
combined defensibly.

## Decision

- **Keyword side: Postgres FTS** on a `GENERATED ALWAYS AS (to_tsvector(...))
  STORED` column with a GIN index. A generated column cannot drift from chunk
  text; same database as vectors means one query surface, transactional
  consistency, and zero new infrastructure. Ranking: `ts_rank_cd` (cover
  density — proximity-aware, suits phrase-like clause queries). Input goes
  through `websearch_to_tsquery`, which is injection-safe and never raises on
  malformed user input.
- **Fusion: RRF** (`score = Σ wᵢ/(60 + rankᵢ)`). Rank-based fusion needs no
  score calibration — dense cosine and ts_rank_cd are on incomparable scales,
  and any interpolation scheme would need per-source normalization that
  drifts with the corpus. rrf_k=60 per Cormack et al.
- **Fuse wide, return narrow**: hybrid retrieves `candidate_k=50` per source,
  fuses, returns top-k. Both sources run concurrently (asyncio.gather).
- `RetrievalMode` enum (dense | keyword | hybrid) on the service — ablation
  axis #2 is a config value.

## Alternatives considered

- **Weighted score interpolation** — requires calibration; rejected above.
- **External BM25 (Elasticsearch/OpenSearch)** — marginally better lexical
  ranking, whole new moving part; against ADR-0001's one-database principle.
- **English stemmer concerns**: legal terms stem acceptably ("indemnification"
  → "indemnif" still matches its family). Defined-term-exact matching could
  use a `simple` config secondary column later if eval shows stemmer damage.

## Consequences

- Exact-term queries now rank the right clause first (verified by integration
  test); paraphrase queries still work via the dense side.
- The eval harness (Ckpt 15) measures precisely what hybrid buys over dense —
  the claim ships with numbers, not vibes.
- FTS index adds ~tens of MB; acceptable within the free-tier budget.
