"""Typed exception hierarchy.

Every layer raises these (or subclasses) — never bare Exception, never
provider SDK exceptions. The API layer (Checkpoint 17) maps them to HTTP
problem+json responses in exactly one place.
"""


class ClausewiseError(Exception):
    """Base class for all application errors."""


class ConfigurationError(ClausewiseError):
    """Invalid or missing configuration detected at startup."""


class IngestionError(ClausewiseError):
    """Failure while parsing, chunking, or persisting contracts."""


class RetrievalError(ClausewiseError):
    """Failure in the retrieval pipeline (vector/keyword search, fusion, rerank)."""


class GenerationError(ClausewiseError):
    """Failure while generating or validating an answer."""


class ProviderError(ClausewiseError):
    """An external provider (embeddings, LLM, reranker) failed after retries.

    Wraps the underlying SDK exception so callers depend only on our types.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
