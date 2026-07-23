# ADR-0004: Embedding model and vector index

- **Status:** accepted
- **Date:** 2026-07-22

## Context

Zero-budget constraint (ADR-0001): no paid embedding APIs, no card. The
evaluation layer also benefits from *deterministic, local* embeddings —
reproducible ablations with no API drift. Corpus: ~50k chunks across two
corpora; Supabase free tier caps the DB at 500MB.

## Decision

- **Model: BAAI/bge-small-en-v1.5** (384-dim, CPU-friendly). Strong
  MTEB-retrieval quality per parameter; 384 dims keeps ~50k vectors around
  ~75MB — comfortably inside the free tier. BGE's query-side instruction
  prefix is applied in `embed_query` only, per the model card.
- **Normalized embeddings**, so cosine distance == dot product; `<=>` with
  `vector_cosine_ops` everywhere.
- **HNSW index** (m=16, ef_construction=64): better query-time recall/latency
  than IVFFlat at our scale, no training step, works incrementally as NULL
  rows get embedded.
- **Vectors as text literals with explicit casts** in SQL — no driver type
  registration; identical behavior across asyncpg local and pooled prod.
- **`embedding_model` stored per row**: model swaps are visible migrations.
  The embed script is resumable via `WHERE embedding IS NULL`.
- Heavy ML deps live in an optional **`ml` dependency group** — CI never
  installs torch; adapters import lazily and fail with actionable messages.

## Alternatives considered

- **OpenAI text-embedding-3-small** — better quality, but paid + card.
  Swappable later behind the port; nothing else changes.
- **bge-base/large (768/1024-dim)** — quality bump not worth 2–3x storage and
  CPU at portfolio scale; revisit if eval shows retrieval is the bottleneck.
- **IVFFlat** — needs list tuning + training on data present at index time;
  HNSW's incremental build fits the resumable-embedding workflow better.

## Consequences

- $0 embedding cost, fully reproducible evals, no rate limits.
- CPU embedding of ~50k chunks takes tens of minutes (one-time; resumable).
- 512-token model window truncates the 0.2% oversized chunks — accepted and
  documented in the chunk report.
