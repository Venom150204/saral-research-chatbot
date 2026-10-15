"""Reciprocal-rank fusion over independently scoped retrieval channels."""

from __future__ import annotations

from dataclasses import dataclass

from .models import SearchHit


@dataclass(frozen=True, slots=True)
class HybridHit:
    rank: int
    rrf_score: float
    hit: SearchHit
    lexical_rank: int | None
    dense_rank: int | None


def reciprocal_rank_fusion(
    lexical: tuple[SearchHit, ...],
    dense: tuple[SearchHit, ...],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
) -> tuple[HybridHit, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    channels = (lexical, dense)
    by_chunk = {}
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    for channel_index, channel in enumerate(channels):
        seen = set()
        for item in channel:
            chunk_id = item.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError("a retrieval channel cannot repeat a chunk")
            seen.add(chunk_id)
            by_chunk[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (
                rank_constant + item.rank
            )
            ranks.setdefault(chunk_id, [None, None])[channel_index] = item.rank
    ordered = sorted(
        by_chunk,
        key=lambda chunk_id: (
            -scores[chunk_id],
            ranks[chunk_id][0] or 10**9,
            ranks[chunk_id][1] or 10**9,
            chunk_id,
        ),
    )
    return tuple(
        HybridHit(
            rank=index,
            rrf_score=scores[chunk_id],
            hit=by_chunk[chunk_id],
            lexical_rank=ranks[chunk_id][0],
            dense_rank=ranks[chunk_id][1],
        )
        for index, chunk_id in enumerate(ordered[:top_k], start=1)
    )
