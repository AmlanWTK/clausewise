# ADR-0003: Chunking strategy

- **Status:** accepted
- **Date:** 2026-07-22

## Context

Retrieval quality lives or dies at chunking. CUAD's data profile shows median
gold spans of ~196 chars inside contracts averaging ~52k chars — retrieval must
resolve clause-sized targets. Fixed token windows routinely cut clauses in half
and glue unrelated clauses together.

## Decision

Two `Chunker` implementations behind the same port:

1. **ClauseAwareChunker (production default).** Walks the parsed section tree;
   each node's own text is a unit tagged with its section path. Units under
   `min_tokens` merge into the adjacent unit (headings alone are noise); units
   over `max_tokens` split at sentence boundaries near `target_tokens`. Chunk
   text carries a `[ARTICLE II > 2.1]` path prefix for embedding context, while
   char offsets always reference the raw canonical span (evaluation depends on
   this separation).
2. **FixedSizeChunker (ablation baseline).** Token windows + overlap, structure
   ignored. Built now, not later, so the Checkpoint 16 ablation compares equals.

Size policy lives in `ChunkerConfig` (target 350 / min 40 / max 480 tokens,
headroom under the embedding model's 512 window). Token counting is an
injectable `TokenCounter`: a chars/4 heuristic today, the real embedding-model
tokenizer once adapters land (Checkpoint 8) — chunker logic never changes.

Chunk ids hash `(contract, strategy, span)` — deterministic, so re-ingestion
is idempotent and strategies never collide.

## Alternatives considered

- **tiktoken counting now** — wrong vocabulary for BGE local embeddings and an
  extra network-fetching dependency; the injectable counter is strictly better.
- **Recursive character splitting (LangChain-style)** — respects paragraphs but
  not legal structure; that's what the baseline approximates anyway.
- **Overlap in the clause-aware chunker** — structure boundaries make overlap
  mostly redundant; revisit only if eval shows boundary misses.

## Consequences

- Eval can map CUAD gold spans onto chunks via pure offset overlap.
- The ablation's chunking axis is a one-line config/DI swap.
- Chunk quality depends on parser quality (96.6% structured-parse rate);
  parser failures degrade to whole-document units, handled by the splitter.
