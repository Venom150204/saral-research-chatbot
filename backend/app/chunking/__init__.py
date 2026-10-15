"""Deterministic provenance-preserving chunk construction."""

from .builder import build_chunks
from .models import (
    SLIDE_PACKING_CONFIG, Chunk, ChunkConfig, ChunkSet, EvidenceReference, location_label,
)

__all__ = [
    "SLIDE_PACKING_CONFIG", "Chunk", "ChunkConfig", "ChunkSet", "EvidenceReference",
    "build_chunks", "location_label",
]
