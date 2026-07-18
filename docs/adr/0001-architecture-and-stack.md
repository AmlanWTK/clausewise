# ADR-0001: Architecture style and core stack

- **Status:** accepted
- **Date:** 2026-07-18

## Context

ClauseWise is a portfolio-grade RAG system over the CUAD legal contract dataset. It must
demonstrate senior-level engineering (clean architecture, formal evaluation, production
concerns) while costing **$0 to build and operate** — no credit card is available, so every
provider and host must have a genuinely free, no-card tier.

## Decision

1. **Clean/hexagonal architecture.** Domain logic (`domain/`) and abstract ports (`ports/`)
   have zero dependencies on provider SDKs, the database, or the web framework. Concrete
   implementations live in `adapters/`. Dependency direction: `adapters → ports → domain`.
2. **No RAG framework.** Retrieval and generation pipelines are hand-built against our own
   interfaces. This shows understanding of the internals and keeps clause-aware chunking,
   hybrid fusion, and evaluation fully under our control.
3. **Zero-cost provider defaults**, all swappable behind ports:
   - Embeddings: local `sentence-transformers` (BGE-small, 384-dim, CPU)
   - Re-ranking: local cross-encoder
   - Generation: Gemini API free tier (no card required)
4. **Single Postgres (16 + pgvector)** for vectors, full-text search, app data, and query
   traces. One database keeps hybrid search transactional and infra minimal.
5. **Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (async) + Alembic, Pydantic v2,
   React + Vite + TS frontend. Tooling: uv, ruff, mypy --strict, pytest, GitHub Actions.
6. **Deployment:** Docker on Hugging Face Spaces (free CPU tier) + Supabase free Postgres.

## Alternatives considered

- **LangChain / LlamaIndex** — faster start, but heavy abstractions fight custom chunking
  and evaluation; reads as tutorial-level to senior reviewers.
- **Dedicated vector DB (Pinecone/Qdrant)** — extra moving part and/or paid tier; pgvector
  is production-appropriate at CUAD scale (≪ 50M vectors).
- **Paid APIs (OpenAI/Anthropic)** — better generation quality, but requires a card.
  Ports make this a config swap later, not a rewrite.

## Consequences

- Easier: testing (fake adapters), ablations (config swaps), provider migration.
- Harder: more up-front interface design; local models add RAM/latency constraints that
  deployment (ADR on topology, Checkpoint 20) must respect.
- Committed to: mypy strict + CI-gated main from the first commit; migrations from day one.
