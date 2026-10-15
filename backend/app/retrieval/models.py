"""Retrieval result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    score: float
    chunk: Chunk

