"""Local ONNX embedding adapter for retrieval."""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, Protocol

from .contracts import EmbeddingBatch


class LocalEmbeddingBackend(Protocol):
    @property
    def embedding_size(self) -> int: ...

    def passage_embed(self, texts: Iterable[str], **kwargs: object) -> Iterable[object]: ...

    def query_embed(self, query: str | Iterable[str], **kwargs: object) -> Iterable[object]: ...


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_query_instruction(model_id: str) -> str | None:
    """The query prefix a model's card prescribes, or None when it has none."""
    if model_id.startswith("BAAI/bge-") and model_id.endswith("-en-v1.5"):
        return BGE_QUERY_INSTRUCTION
    return None


class FastEmbedder:
    """Provider-neutral adapter around FastEmbed's local ONNX runtime."""

    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-base-en-v1.5",
        dimensions: int = 768,
        backend: LocalEmbeddingBackend | None = None,
        cache_dir: str | None = ".cache/fastembed",
        batch_size: int = 32,
        timeout_seconds: float = 120.0,
        query_instruction: str | None = None,
    ) -> None:
        if dimensions < 1 or batch_size < 1 or timeout_seconds <= 0:
            raise ValueError("embedding dimensions, batch size, and timeout must be positive")
        self._model_id = model_id
        self._dimensions = dimensions
        self._backend = backend
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self._query_instruction = query_instruction

    def _get_backend(self) -> LocalEmbeddingBackend:
        if self._backend is None:
            from fastembed import TextEmbedding

            self._backend = TextEmbedding(
                model_name=self._model_id,
                cache_dir=self._cache_dir,
                providers=["CPUExecutionProvider"],
            )
        if self._backend.embedding_size != self._dimensions:
            raise ValueError(
                f"local embedding model has {self._backend.embedding_size} dimensions; "
                f"the index requires {self._dimensions}"
            )
        return self._backend

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        return await self._embed(texts, query=False)

    async def embed_query(self, text: str) -> EmbeddingBatch:
        return await self._embed((text,), query=True)

    async def _embed(self, texts: tuple[str, ...], *, query: bool) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-empty")
        started = time.perf_counter()

        def encode() -> tuple[tuple[float, ...], ...]:
            backend = self._get_backend()
            queries = (
                tuple(self._query_instruction + text for text in texts)
                if query and self._query_instruction
                else texts
            )
            generated = (
                backend.query_embed(queries, batch_size=self._batch_size)
                if query
                else backend.passage_embed(texts, batch_size=self._batch_size)
            )
            vectors = tuple(tuple(float(value) for value in vector) for vector in generated)
            if len(vectors) != len(texts):
                raise ValueError("local embedding model returned an unexpected vector count")
            if any(len(vector) != self._dimensions for vector in vectors):
                raise ValueError("local embedding model returned an unexpected dimension")
            return vectors

        try:
            vectors = await asyncio.wait_for(
                asyncio.to_thread(encode), timeout=self._timeout_seconds
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"local embedding batch exceeded {self._timeout_seconds:g} seconds"
            ) from error
        return EmbeddingBatch(
            model_id=self._model_id,
            dimensions=self._dimensions,
            vectors=vectors,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
