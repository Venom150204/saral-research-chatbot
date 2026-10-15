"""Gemini Embedding 2 adapter with explicit retrieval instructions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from google import genai
from google.genai import types

from .contracts import EmbeddingBatch


@dataclass(frozen=True, slots=True)
class TransportEmbedding:
    vector: tuple[float, ...]
    input_tokens: int | None = None


class EmbeddingTransport(Protocol):
    async def embed(
        self, *, model_id: str, text: str, dimensions: int
    ) -> TransportEmbedding: ...


class GoogleGenAIEmbeddingTransport:
    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def embed(
        self, *, model_id: str, text: str, dimensions: int
    ) -> TransportEmbedding:
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        async with client.aio as async_client:
            response = await async_client.models.embed_content(
                model=model_id,
                contents=types.Content(
                    role="user", parts=[types.Part.from_text(text=text)]
                ),
                config=types.EmbedContentConfig(output_dimensionality=dimensions),
            )
        if not response.embeddings or response.embeddings[0].values is None:
            raise ValueError("Gemini returned no embedding vector")
        vector = tuple(float(value) for value in response.embeddings[0].values)
        metadata = getattr(response, "usage_metadata", None)
        return TransportEmbedding(
            vector=vector,
            input_tokens=getattr(metadata, "prompt_token_count", None),
        )


class GeminiEmbedding2:
    def __init__(
        self,
        *,
        transport: EmbeddingTransport | None = None,
        api_key: str | None = None,
        model_id: str = "gemini-embedding-2",
        dimensions: int = 768,
        max_concurrency: int = 4,
        batch_timeout_seconds: float = 60.0,
    ) -> None:
        if dimensions not in {768, 1536, 3072}:
            raise ValueError("use a documented recommended embedding dimension")
        self._transport = transport or GoogleGenAIEmbeddingTransport(api_key=api_key)
        self._model_id = model_id
        self._dimensions = dimensions
        if max_concurrency < 1 or batch_timeout_seconds <= 0:
            raise ValueError("embedding concurrency and timeout must be positive")
        self._max_concurrency = max_concurrency
        self._batch_timeout_seconds = batch_timeout_seconds

    async def embed_documents(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        return await self._embed(texts, kind="document")

    async def embed_query(self, text: str) -> EmbeddingBatch:
        return await self._embed((text,), kind="query")

    async def _embed(
        self, texts: tuple[str, ...], *, kind: Literal["document", "query"]
    ) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-empty")
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def embed_one(text: str) -> TransportEmbedding:
            instructed = f"task: retrieval | {kind}: {text}"
            async with semaphore:
                return await self._transport.embed(
                    model_id=self._model_id,
                    text=instructed,
                    dimensions=self._dimensions,
                )

        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*(embed_one(text) for text in texts)),
                timeout=self._batch_timeout_seconds,
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"embedding batch exceeded {self._batch_timeout_seconds:g} seconds"
            ) from error
        vectors = []
        token_count = 0
        for response in responses:
            if len(response.vector) != self._dimensions:
                raise ValueError("Gemini returned an unexpected embedding dimension")
            vectors.append(response.vector)
            token_count += response.input_tokens or 0
        return EmbeddingBatch(
            model_id=self._model_id,
            dimensions=self._dimensions,
            vectors=tuple(vectors),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=token_count or None,
        )
