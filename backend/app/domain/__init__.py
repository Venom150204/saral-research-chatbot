"""Domain models shared across SARAL pipeline components."""

from .source import (
    BoundingBox,
    ExtractedDocument,
    ExtractionWarning,
    SourceBlock,
    SourceLocation,
    VisualElement,
)

__all__ = [
    "BoundingBox",
    "ExtractedDocument",
    "ExtractionWarning",
    "SourceBlock",
    "SourceLocation",
    "VisualElement",
]
