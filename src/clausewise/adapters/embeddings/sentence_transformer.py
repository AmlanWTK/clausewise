"""Local embedding provider: sentence-transformers (BGE family), $0 per token.

Implements the EmbeddingProvider port. The heavyweight import happens inside
``__init__`` so the module can be imported (e.g. by CI collecting tests)
without torch installed; instantiating without the ml group raises a clear
ConfigurationError instead of an ImportError five frames deep.
"""

import asyncio
from collections.abc import Sequence

from clausewise.config import Settings
from clausewise.domain import EmbeddingBatch, Vector
from clausewise.domain.errors import ConfigurationError, ProviderError

# BGE models are trained with an instruction prefix on the *query* side only.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbeddingProvider:
    """CPU-friendly local embeddings; normalized vectors (cosine-ready)."""

    def __init__(self, settings: Settings, *, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = "sentence-transformers not installed — run: uv sync --group ml"
            raise ConfigurationError(msg) from exc

        self._model_name = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = batch_size
        self._model = SentenceTransformer(self._model_name, device="cpu")

        actual = self._model.get_sentence_embedding_dimension()
        if actual != self._dimensions:
            msg = (
                f"Model {self._model_name} produces {actual}-dim vectors but config "
                f"says {self._dimensions} — embedding column would be corrupted."
            )
            raise ConfigurationError(msg)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        # CPU-bound work off the event loop.
        return await asyncio.to_thread(self._encode_documents, list(texts))

    async def embed_query(self, text: str) -> Vector:
        batch = await asyncio.to_thread(self._encode_documents, [_QUERY_PREFIX + text])
        return batch.vectors[0]

    def _encode_documents(self, texts: list[str]) -> EmbeddingBatch:
        try:
            arrays = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,  # cosine == dot product downstream
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            token_ids = self._model.tokenizer(texts, truncation=True)["input_ids"]
        except Exception as exc:
            raise ProviderError("sentence-transformers", str(exc)) from exc
        return EmbeddingBatch(
            vectors=tuple(tuple(float(x) for x in row) for row in arrays),
            model=self._model_name,
            dimensions=self._dimensions,
            total_tokens=sum(len(ids) for ids in token_ids),
        )
