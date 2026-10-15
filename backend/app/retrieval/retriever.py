"""Two-channel retrieval inside one document version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.app.chunking import Chunk, ChunkSet
from backend.app.normalization import to_written_form

from .dense import DenseIndex
from .hybrid import HybridHit, reciprocal_rank_fusion
from .lexical import LexicalIndex


class RetrievalEmbedder(Protocol):
    async def embed_query(self, text: str): ...

    async def embed_documents(self, texts: tuple[str, ...]): ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """A bullet that may support a sentence, with its rank in each channel."""

    chunk: Chunk
    keyword_rank: int | None
    embedding_rank: int | None


class DeckRetriever:
    def __init__(
        self,
        chunk_set: ChunkSet,
        vectors: Sequence[Sequence[float]],
        *,
        embedder: RetrievalEmbedder,
        embedding_model_id: str,
    ) -> None:
        self._chunk_set = chunk_set
        self._keyword = LexicalIndex(chunk_set)
        self._embeddings = DenseIndex(chunk_set, vectors)
        self._embedder = embedder
        self._embedding_model_id = embedding_model_id

    async def lookup(self, question: str, *, top_k: int = 5) -> tuple[HybridHit, ...]:
        """Rank bullets for a question, fusing both channels by reciprocal rank."""
        written = to_written_form(question)
        depth = min(max(top_k * 3, 10), len(self._chunk_set.chunks))
        batch = await self._embedder.embed_query(written)
        return reciprocal_rank_fusion(
            self._keyword_hits(written, depth),
            self._embedding_hits(batch, depth),
            top_k=top_k,
        )

    async def candidates(
        self, sentence: str, *, per_channel: int = 3
    ) -> tuple[Candidate, ...]:
        """Pool the best bullets of each channel as possible support for a sentence."""
        if per_channel < 1:
            raise ValueError("per_channel must be positive")
        written = to_written_form(sentence)
        batch = await self._embedder.embed_documents((written,))
        keyword = self._keyword_hits(written, per_channel)
        embedding = self._embedding_hits(batch, per_channel)
        ranks: dict[str, list[int | None]] = {}
        chunks: dict[str, Chunk] = {}
        for channel, hits in enumerate((keyword, embedding)):
            for hit in hits:
                chunks[hit.chunk.chunk_id] = hit.chunk
                ranks.setdefault(hit.chunk.chunk_id, [None, None])[channel] = hit.rank
        ordered = sorted(
            chunks,
            key=lambda chunk_id: (
                min(rank for rank in ranks[chunk_id] if rank is not None),
                ranks[chunk_id][0] is None,
                chunks[chunk_id].ordinal,
            ),
        )
        return tuple(
            Candidate(chunks[chunk_id], ranks[chunk_id][0], ranks[chunk_id][1])
            for chunk_id in ordered
        )

    def _keyword_hits(self, written: str, depth: int):
        return self._keyword.search(
            written,
            document_id=self._chunk_set.document_id,
            document_version=self._chunk_set.document_version,
            top_k=depth,
        )

    def _embedding_hits(self, batch, depth: int):
        if batch.model_id != self._embedding_model_id:
            raise ValueError("query and document embeddings use different models")
        return self._embeddings.search(
            batch.vectors[0],
            document_id=self._chunk_set.document_id,
            document_version=self._chunk_set.document_version,
            top_k=depth,
        )
