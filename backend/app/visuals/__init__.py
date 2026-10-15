"""Provider-neutral visual-description contracts."""

from .contracts import (
    VisualDescription,
    VisualDescriptionRequest,
    VisualDescriptionResult,
    VisionDescriber,
    build_description_requests,
)
from .enrichment import (
    DerivedVisualEvidence,
    VisualEnrichmentIssue,
    VisualEnrichmentResult,
    VisualEnrichmentTelemetry,
    enrich_document_visuals,
    select_visuals_for_description,
)

__all__ = [
    "VisualDescription",
    "VisualDescriptionRequest",
    "VisualDescriptionResult",
    "VisionDescriber",
    "build_description_requests",
    "DerivedVisualEvidence",
    "VisualEnrichmentIssue",
    "VisualEnrichmentResult",
    "VisualEnrichmentTelemetry",
    "enrich_document_visuals",
    "select_visuals_for_description",
]
