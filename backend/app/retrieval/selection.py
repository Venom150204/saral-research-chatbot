"""Evidence selection policy: the whole source when it fits, retrieval when it does not."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.chunking import Chunk, ChunkSet

from .hybrid import HybridHit


DEFAULT_EVIDENCE_CHARACTER_BUDGET = 60_000


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    mode: str
    chunks: tuple[Chunk, ...]
    total_characters: int


def document_fits_evidence_budget(
    chunk_set: ChunkSet,
    *,
    max_characters: int = DEFAULT_EVIDENCE_CHARACTER_BUDGET,
    max_chunks: int | None = None,
) -> bool:
    """True when every chunk can be passed, so no retrieval query is needed."""
    if max_chunks is not None and len(chunk_set.chunks) > max_chunks:
        return False
    return sum(len(chunk.text) for chunk in chunk_set.chunks) <= max_characters


def select_generation_evidence(
    chunk_set: ChunkSet,
    ranked_hits: tuple[HybridHit, ...],
    *,
    max_characters: int = DEFAULT_EVIDENCE_CHARACTER_BUDGET,
    max_chunks: int | None = None,
    relevance_share: float = 0.7,
) -> EvidenceSelection:
    """Pass the whole document when it fits; otherwise mix relevance and coverage."""
    if max_characters < 1 or (max_chunks is not None and max_chunks < 1):
        raise ValueError("evidence budgets must be positive")
    if not 0 <= relevance_share <= 1:
        raise ValueError("relevance_share must be between zero and one")
    ordered = tuple(sorted(chunk_set.chunks, key=lambda chunk: chunk.ordinal))
    total = sum(len(chunk.text) for chunk in ordered)
    if document_fits_evidence_budget(
        chunk_set, max_characters=max_characters, max_chunks=max_chunks
    ):
        return EvidenceSelection("whole_document_coverage", ordered, total)

    selected: dict[str, Chunk] = {}
    used = 0

    def add(chunk: Chunk, limit: int) -> None:
        nonlocal used
        if chunk.chunk_id in selected:
            return
        if max_chunks is not None and len(selected) >= max_chunks:
            return
        if used + len(chunk.text) > limit:
            return
        selected[chunk.chunk_id] = chunk
        used += len(chunk.text)

    relevance_limit = int(max_characters * relevance_share) if ranked_hits else 0
    relevance_slots = None if max_chunks is None else max(1, round(max_chunks * relevance_share))
    for hit in ranked_hits:
        if relevance_slots is not None and len(selected) >= relevance_slots:
            break
        add(hit.hit.chunk, relevance_limit)

    step = len(ordered)
    while step >= 1:
        for index in range(0, len(ordered), step):
            add(ordered[index], max_characters)
        add(ordered[-1], max_characters)
        if step == 1:
            break
        step //= 2
    return EvidenceSelection(
        "hybrid_relevance_plus_structural_coverage"
        if ranked_hits else "structural_coverage_only",
        tuple(sorted(selected.values(), key=lambda chunk: chunk.ordinal)),
        used,
    )
