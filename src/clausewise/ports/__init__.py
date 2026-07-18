"""Ports: abstract interfaces the application core depends on.

Adapters (``clausewise.adapters``) implement these; services depend only on
the Protocols. Rules:
- Ports use domain types exclusively — no provider SDK types in signatures.
- All I/O-bound ports are async.
"""

from clausewise.ports.chunker import Chunker
from clausewise.ports.embeddings import EmbeddingProvider
from clausewise.ports.keyword_index import KeywordIndex
from clausewise.ports.llm import LLMProvider
from clausewise.ports.reranker import Reranker
from clausewise.ports.vector_store import VectorStore

__all__ = [
    "Chunker",
    "EmbeddingProvider",
    "KeywordIndex",
    "LLMProvider",
    "Reranker",
    "VectorStore",
]
