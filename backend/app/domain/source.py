"""Canonical provenance models for slide decks, summaries and papers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A shape position in English Metric Units (EMU)."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("bounding-box width and height must be non-negative")


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """An extractor-established location: a slide, a PDF page, or source lines."""

    slide_number: int | None = None
    shape_id: int | None = None
    paragraph_index: int | None = None
    table_row: int | None = None
    table_column: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    bounding_box: BoundingBox | None = None
    page_number: int | None = None

    def __post_init__(self) -> None:
        if self.slide_number is not None and self.slide_number < 1:
            raise ValueError("slide_number must be one-based")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if self.line_start is not None and self.line_start < 1:
            raise ValueError("line_start must be one-based")
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end cannot precede line_start")


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """A stable unit of original extracted text and its provenance."""

    block_id: str
    parent_id: str
    ordinal: int
    text: str
    location: SourceLocation
    block_type: str
    hierarchy_level: int | None = None

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id is required")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.text.strip():
            raise ValueError("source blocks cannot contain only whitespace")


@dataclass(frozen=True, slots=True)
class ExtractionWarning:
    """A machine-readable extraction limitation that remains visible downstream."""

    code: str
    message: str
    slide_number: int | None = None


@dataclass(frozen=True, slots=True)
class VisualElement:
    """A visual placeholder retained even when its meaning is not interpreted."""

    visual_id: str
    parent_id: str
    slide_number: int
    shape_id: int
    shape_name: str
    visual_type: str
    bounding_box: BoundingBox
    alt_text: str | None = None
    asset_hash: str | None = None
    media_type: str | None = None
    byte_size: int | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None
    asset_uri: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Versioned extraction result used as the input to future chunking."""

    document_id: str
    document_version: str
    content_hash: str
    filename: str
    media_type: str
    source_kind: str
    parser_name: str
    parser_version: str
    unit_count: int
    blocks: tuple[SourceBlock, ...]
    visuals: tuple[VisualElement, ...] = ()
    warnings: tuple[ExtractionWarning, ...] = ()

    def __post_init__(self) -> None:
        if self.unit_count < 1:
            raise ValueError("unit_count must be positive")
        if not self.blocks:
            raise ValueError("an extracted document requires at least one source block")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without changing source text."""

        return asdict(self)
