"""In-memory fake adapters implementing every port.

These make all services testable with zero network, zero Docker, zero cost.
They are deliberately simple but honest: deterministic embeddings, real cosine
math, real ranking — so tests exercise actual logic, not mocks of it.
"""

import hashlib
import math
from collections.abc import Sequence

from clausewise.domain import (
    Chunk,
    ChunkMetadata,
    Contract,
    EmbeddingBatch,
    LLMResponse,
    RetrievedChunk,
    Section,
    TokenUsage,
    Vector,
)
from clausewise.domain.retrieval import RetrievalSource


def _deterministic_vector(text: str, dimensions: int) -> Vector:
    """Map text to a stable pseudo-embedding via hashing. Same text → same vector."""
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        values.extend(b / 255.0 - 0.5 for b in digest)
        counter += 1
    norm = math.sqrt(sum(v * v for v in values[:dimensions]))
    return tuple(v / norm for v in values[:dimensions])


def cosine_similarity(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeEmbeddingProvider:
    """Deterministic hash-based embeddings."""

    def __init__(self, dimensions: int = 32) -> None:
        self._dimensions = dimensions
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embed-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls += 1
        return EmbeddingBatch(
            vectors=tuple(_deterministic_vector(t, self._dimensions) for t in texts),
            model=self.model_name,
            dimensions=self._dimensions,
            total_tokens=sum(len(t.split()) for t in texts),
        )

    async def embed_query(self, text: str) -> Vector:
        return _deterministic_vector(text, self._dimensions)


class FakeLLMProvider:
    """Returns canned responses in order; records every prompt it sees."""

    def __init__(self, responses: Sequence[str] = ("fake answer",)) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-llm-v1"

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        text = self._responses[min(len(self.prompts) - 1, len(self._responses) - 1)]
        return LLMResponse(
            text=text,
            model=self.model_name,
            usage=TokenUsage(input_tokens=len(prompt.split()), output_tokens=len(text.split())),
        )


class InMemoryVectorStore:
    """Real cosine-similarity search over an in-memory corpus dict."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, tuple[Chunk, Vector]]] = {}

    async def upsert(
        self,
        corpus: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        embedding_model: str,
    ) -> int:
        store = self._data.setdefault(corpus, {})
        for chunk, vector in zip(chunks, vectors, strict=True):
            store[chunk.id] = (chunk, vector)
        return len(chunks)

    async def search(
        self,
        corpus: str,
        query_vector: Vector,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        candidates = [
            (chunk, vector)
            for chunk, vector in self._data.get(corpus, {}).values()
            if contract_ids is None or chunk.contract_id in contract_ids
        ]
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=cosine_similarity(query_vector, vector),
                source=RetrievalSource.DENSE,
            )
            for chunk, vector in candidates
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]


class InMemoryKeywordIndex:
    """Naive term-overlap ranking — a stand-in for Postgres FTS."""

    def __init__(self) -> None:
        self._data: dict[str, list[Chunk]] = {}

    async def index(self, corpus: str, chunks: Sequence[Chunk]) -> int:
        self._data.setdefault(corpus, []).extend(chunks)
        return len(chunks)

    async def search(
        self,
        corpus: str,
        query: str,
        *,
        k: int = 10,
        contract_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        terms = {t.lower() for t in query.split()}
        results: list[RetrievedChunk] = []
        for chunk in self._data.get(corpus, []):
            if contract_ids is not None and chunk.contract_id not in contract_ids:
                continue
            chunk_terms = {t.lower() for t in chunk.text.split()}
            overlap = len(terms & chunk_terms)
            if overlap:
                results.append(
                    RetrievedChunk(
                        chunk=chunk, score=float(overlap), source=RetrievalSource.KEYWORD
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)[:k]


class FakeReranker:
    """Reranks by term overlap — enough to test that ordering changes."""

    @property
    def model_name(self) -> str:
        return "fake-rerank-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        terms = {t.lower() for t in query.split()}
        rescored = [
            RetrievedChunk(
                chunk=c.chunk,
                score=float(len(terms & {t.lower() for t in c.chunk.text.split()})),
                source=RetrievalSource.RERANKED,
            )
            for c in candidates
        ]
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]


class FakeChunker:
    """One chunk per section; falls back to whole-contract chunk."""

    @property
    def name(self) -> str:
        return "fake"

    def chunk(self, contract: Contract, sections: Sequence[Section]) -> list[Chunk]:
        if not sections:
            return [
                Chunk(
                    id=f"{contract.id}:0",
                    contract_id=contract.id,
                    text=contract.text,
                    char_start=0,
                    char_end=len(contract.text),
                    token_count=len(contract.text.split()),
                    metadata=ChunkMetadata(contract_title=contract.title),
                )
            ]
        return [
            Chunk(
                id=f"{contract.id}:{i}",
                contract_id=contract.id,
                text=contract.text[s.char_start : s.char_end],
                char_start=s.char_start,
                char_end=s.char_end,
                token_count=len(contract.text[s.char_start : s.char_end].split()),
                metadata=ChunkMetadata(contract_title=contract.title, section_path=(s.number,)),
            )
            for i, s in enumerate(sections)
        ]
