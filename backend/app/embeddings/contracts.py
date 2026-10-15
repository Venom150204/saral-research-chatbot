"""Provider-neutral embedding boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    model_id: str
    dimensions: int
    vectors: tuple[tuple[float, ...], ...]
    latency_ms: float
    input_tokens: int | None = None


class Embedder(Protocol):
    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch: ...

    async def embed_query(self, text: str) -> EmbeddingBatch: ...
