# ADR-0002: Ports & provider abstraction

- **Status:** accepted
- **Date:** 2026-07-18

## Context

The system depends on external capabilities (embeddings, LLM, vector search,
lexical search, re-ranking) whose providers we want to swap freely: local
models for $0 evals, free-tier APIs for generation, paid APIs later without a
rewrite. The evaluation layer (ablations) also demands that pipeline pieces be
interchangeable via configuration.

## Decision

- Interfaces are `typing.Protocol` classes in `clausewise/ports/`, one file per
  capability: `EmbeddingProvider`, `LLMProvider`, `VectorStore`, `KeywordIndex`,
  `Reranker`, `Chunker`.
- **Structural typing over inheritance:** adapters don't subclass anything —
  they just match the shape. mypy `--strict` verifies conformance at type-check
  time; `@runtime_checkable` + contract tests verify it at test time.
- Ports speak **domain types only** (`Chunk`, `RetrievedChunk`, `EmbeddingBatch`,
  `LLMResponse`). Provider SDK types and exceptions never cross the boundary;
  failures surface as `ProviderError`.
- Results carry provenance and accounting by design: scores are tagged with a
  `RetrievalSource` (scores are not comparable across stages), every LLM/embedding
  result carries token counts (cost accounting is not an afterthought).
- Full in-memory fakes live in `tests/fakes.py` and are exercised by contract
  tests — every service can be tested with zero network and zero cost.

## Alternatives considered

- **ABC base classes** — nominal typing forces adapters to import the core;
  Protocols keep adapters decoupled and are the modern Python idiom.
- **A single "Provider" god-interface** — violates interface segregation; ports
  are split per capability so an adapter implements exactly what it is.

## Consequences

- Ablations (Ckpt 16) become dependency-injection swaps, not code changes.
- Adding a provider = one adapter file + contract tests; nothing else moves.
- Discipline required: the temptation to "just import the SDK type" in a
  service is rejected in review.
