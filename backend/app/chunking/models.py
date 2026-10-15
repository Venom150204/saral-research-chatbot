"""Canonical retrieval chunk models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    version: str = "bullet-v1"
    unit: Literal["bullet", "slide"] = "bullet"
    target_chars: int = 1_200
    summary_strategy: Literal["section_paragraph", "recursive_structural"] = (
        "section_paragraph"
    )
    filter_repeated_slide_boilerplate: bool = True
    table_layout: Literal["row", "grouped"] = "grouped"

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("chunk configuration version is required")
        if self.unit not in {"bullet", "slide"}:
            raise ValueError("unsupported chunk unit")
        if self.target_chars < 100:
            raise ValueError("target_chars must be at least 100")
        if self.table_layout not in {"row", "grouped"}:
            raise ValueError("unsupported table chunk layout")
        if self.summary_strategy not in {
            "section_paragraph",
            "recursive_structural",
        }:
            raise ValueError("unsupported summary chunking strategy")


SLIDE_PACKING_CONFIG = ChunkConfig(
    version="slide-aware-v2", unit="slide", table_layout="row"
)


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    evidence_kind: Literal["source_block", "derived_visual"]
    slide_number: int | None = None
    shape_id: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    page_number: int | None = None


def location_label(evidence: tuple[EvidenceReference, ...] | list) -> str | None:
    """Where a unit comes from, in the words a reader uses: "slide 4", "page 3",
    "lines 12-18". None when the extractor established no location."""
    for ref in evidence:
        if ref.slide_number is not None:
            return f"slide {ref.slide_number}"
    for ref in evidence:
        if ref.page_number is not None:
            return f"page {ref.page_number}"
    for ref in evidence:
        if ref.line_start is not None:
            end = ref.line_end if ref.line_end not in (None, ref.line_start) else None
            return f"lines {ref.line_start}-{end}" if end else f"line {ref.line_start}"
    return None


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_version: str
    parent_id: str
    ordinal: int
    chunk_type: str
    evidence_kind: Literal["native", "derived"]
    text: str
    search_text: str
    evidence: tuple[EvidenceReference, ...]
    config_version: str
    chunking_strategy: str
    supports_quantitative_claims: bool
    oversized: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.search_text.strip():
            raise ValueError("chunk text cannot be empty")
        if not self.evidence:
            raise ValueError("chunks require at least one evidence reference")
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChunkSet:
    document_id: str
    document_version: str
    config: ChunkConfig
    chunks: tuple[Chunk, ...]
    index_version: str = "legacy"

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("chunk set cannot be empty")
        if not self.index_version.strip():
            raise ValueError("chunk set requires an index version")
