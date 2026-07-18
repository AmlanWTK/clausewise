# Architecture

ClauseWise follows a clean/hexagonal architecture. Dependencies point inward:
adapters and entry points depend on ports; ports depend on the domain; the
domain depends on nothing.

```mermaid
flowchart TB
    subgraph entry["Entry points"]
        API["FastAPI app (Ckpt 17)"]
        CLI["CLI scripts: ingest / query / evaluate"]
    end

    subgraph services["Application services"]
        ING["Ingestion pipeline (Ckpt 5–7)"]
        RET["Retrieval pipeline (Ckpt 9–12)"]
        GEN["Generation service (Ckpt 13–14)"]
        EVAL["Evaluation harness (Ckpt 15–16)"]
    end

    subgraph ports["Ports (Protocols)"]
        P1["EmbeddingProvider"]
        P2["LLMProvider"]
        P3["VectorStore"]
        P4["KeywordIndex"]
        P5["Reranker"]
        P6["Chunker"]
    end

    subgraph domain["Domain (pure, dependency-free)"]
        D1["Contract · Section · Chunk"]
        D2["RetrievedChunk · EmbeddingBatch"]
        D3["Answer · Citation · Refusal"]
        D4["Typed errors"]
    end

    subgraph adapters["Adapters"]
        A1["sentence-transformers (local, $0)"]
        A2["Gemini API (free tier)"]
        A3["pgvector / Postgres FTS"]
        A4["cross-encoder (local, $0)"]
        A5["tests/fakes.py (in-memory)"]
    end

    entry --> services
    services --> ports
    ports --> domain
    adapters -. implement .-> ports
```

## Rules

1. `domain/` imports nothing outside the standard library.
2. `ports/` imports only `domain/`.
3. Services import ports + domain, never adapters — adapters are injected at
   entry points (FastAPI dependencies, CLI wiring).
4. Provider SDK types and exceptions never cross a port boundary
   (`ProviderError` wraps them).
5. Scores carry `RetrievalSource` provenance; scores from different stages are
   never compared directly.

## Why it matters here

The ablation study (Checkpoint 16) — the project's flagship artifact — is only
cheap because chunkers, retrievers, and rerankers are injectable: each ablation
cell is a different wiring of the same services, not a code branch.
