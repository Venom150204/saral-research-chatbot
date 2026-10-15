"""In-memory cosine retrieval for a measured dense baseline."""

from __future__ import annotations

import math
from typing import Sequence

from backend.app.chunking import ChunkSet

from .models import SearchHit
from .scope import validate_scope


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vectors require the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine vectors cannot have zero magnitude")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


class DenseIndex:
    def __init__(self, chunk_set: ChunkSet, vectors: Sequence[Sequence[float]]) -> None:
        if len(vectors) != len(chunk_set.chunks):
            raise ValueError("each chunk requires exactly one embedding")
        if not vectors or not vectors[0]:
            raise ValueError("dense index requires non-empty embeddings")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("all embeddings must have the same dimension")
        self._chunk_set = chunk_set
        self._vectors = tuple(tuple(vector) for vector in vectors)
        self.dimension = dimension

    def search(
        self,
        query_vector: Sequence[float],
        *,
        document_id: str,
        document_version: str,
        top_k: int = 5,
        slide_number: int | None = None,
    ) -> tuple[SearchHit, ...]:
        validate_scope(
            self._chunk_set,
            document_id=document_id,
            document_version=document_version,
        )
        if len(query_vector) != self.dimension:
            raise ValueError("query embedding dimension does not match the index")
        scored = []
        for index, (chunk, vector) in enumerate(
            zip(self._chunk_set.chunks, self._vectors, strict=True)
        ):
            if slide_number is not None and not any(
                ref.slide_number == slide_number for ref in chunk.evidence
            ):
                continue
            scored.append((cosine_similarity(query_vector, vector), index, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            SearchHit(rank=rank, score=score, chunk=chunk)
            for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1)
        )
