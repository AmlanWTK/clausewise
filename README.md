# ClauseWise

**Production-grade Retrieval-Augmented Generation for legal contract Q&A.**

Ask questions about real commercial contracts and get answers that **cite the exact source clauses** — or an honest *"not found in the provided contracts"* instead of a hallucination.

> 🚧 In active development. Evaluation results and live demo link will land here as checkpoints complete.

## Why this isn't another RAG tutorial

| Standard tutorial | ClauseWise |
|---|---|
| Random PDF, no ground truth | [CUAD](https://www.atticusprojectai.org/cuad): 510 real contracts, 13k+ lawyer-annotated clauses |
| Fixed-token chunking | Clause-aware chunking that follows legal structure |
| Vector-only similarity | Hybrid retrieval (dense + keyword) with cross-encoder re-ranking |
| Free-form generation | Verified citations, confidence scores, explicit refusal path |
| "It works when I tried it" | Precision / recall / MRR on a frozen test set + full ablation study |
| Local notebook | Deployed, observable, rate-limited public API |

## Architecture

Clean/hexagonal architecture: domain logic depends on abstract ports; providers (embeddings, LLM, vector store, reranker) are swappable adapters. Full diagram in `docs/` (Checkpoint 3).

```
ingestion → clause-aware chunking → dedup → pgvector + FTS
query → hybrid retrieval (RRF) → cross-encoder rerank → grounded generation → cited answer | refusal
```

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev        # install everything
uv run pre-commit install  # enable commit hooks
uv run pytest              # run tests
uv run ruff check . && uv run mypy   # lint + typecheck
```

## Dataset attribution

This project uses **CUAD v1** (Contract Understanding Atticus Dataset) by **The Atticus Project, Inc.**, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Contracts are sourced from public US SEC EDGAR filings.

## License

MIT — see [LICENSE](LICENSE).
