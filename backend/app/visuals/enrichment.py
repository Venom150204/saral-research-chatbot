"""Orchestrate selective visual description while preserving source provenance."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

from backend.app.domain import ExtractedDocument
from backend.app.storage import AssetStore

from .contracts import VisionDescriber, build_description_requests


@dataclass(frozen=True, slots=True)
class DerivedVisualEvidence:
    """Model-derived evidence linked to the immutable source visual."""

    evidence_id: str
    document_id: str
    document_version: str
    visual_id: str
    asset_hash: str
    slide_number: int
    shape_id: int
    model_id: str
    prompt_version: str
    literal_description: str
    visible_text: tuple[str, ...]
    entities: tuple[str, ...]
    relationships: tuple[str, ...]
    uncertainty: str | None
    quantitative_claims_require_corroboration: bool = True
    derived: bool = True

    def __post_init__(self) -> None:
        if not self.asset_hash:
            raise ValueError("derived visual evidence requires an asset hash")
        if not self.derived:
            raise ValueError("visual enrichment must remain marked as derived")
        if not self.quantitative_claims_require_corroboration:
            raise ValueError("VLM-only quantitative claims require corroboration")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualEnrichmentTelemetry:
    visual_id: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class VisualEnrichmentIssue:
    """A picture that produced no description, with the observable reason."""

    visual_id: str
    slide_number: int
    status: Literal["skipped", "failed"]
    reason: str


@dataclass(frozen=True, slots=True)
class VisualEnrichmentResult:
    document_id: str
    document_version: str
    evidence: tuple[DerivedVisualEvidence, ...]
    telemetry: tuple[VisualEnrichmentTelemetry, ...]
    issues: tuple[VisualEnrichmentIssue, ...] = ()


MIN_CONTENT_PIXELS = 48


def select_visuals_for_description(
    document: ExtractedDocument,
    *,
    skip_decorative: bool,
    max_visuals: int | None,
) -> tuple[tuple[str, ...], tuple[VisualEnrichmentIssue, ...]]:
    """Choose which pictures are worth a model call, and say why the rest are not."""
    pictures = [
        visual for visual in document.visuals
        if visual.visual_type == "picture" and visual.asset_hash is not None
    ]
    slides_by_asset: dict[str, set[int]] = {}
    for visual in pictures:
        slides_by_asset.setdefault(visual.asset_hash, set()).add(visual.slide_number)
    repeated_on = max(3, (4 * document.unit_count + 4) // 5)
    selected: list[str] = []
    issues: list[VisualEnrichmentIssue] = []
    for visual in pictures:
        reason = None
        if skip_decorative:
            sides = [
                side for side in (visual.pixel_width, visual.pixel_height)
                if side is not None
            ]
            if sides and min(sides) < MIN_CONTENT_PIXELS:
                reason = f"smaller than {MIN_CONTENT_PIXELS} pixels on one side"
            elif len(slides_by_asset[visual.asset_hash]) >= repeated_on:
                reason = "same image repeated on most slides"
        if reason is None and max_visuals is not None and len(selected) >= max_visuals:
            reason = f"over the limit of {max_visuals} described images per document"
        if reason is None:
            selected.append(visual.visual_id)
        else:
            issues.append(
                VisualEnrichmentIssue(
                    visual.visual_id, visual.slide_number, "skipped", reason
                )
            )
    return tuple(selected), tuple(issues)


async def enrich_document_visuals(
    document: ExtractedDocument,
    *,
    asset_store: AssetStore,
    describer: VisionDescriber,
    max_slide_context_chars: int = 8_000,
    continue_on_error: bool = False,
    skip_decorative: bool = False,
    max_visuals: int | None = None,
    max_concurrency: int = 1,
) -> VisualEnrichmentResult:
    """Describe pictures and link each description to its immutable source image."""

    if max_slide_context_chars < 1:
        raise ValueError("max_slide_context_chars must be positive")
    if max_concurrency < 1 or (max_visuals is not None and max_visuals < 0):
        raise ValueError("visual description limits must be positive")
    contexts = _build_slide_context(document, max_chars=max_slide_context_chars)
    selected_ids, issues_found = select_visuals_for_description(
        document, skip_decorative=skip_decorative, max_visuals=max_visuals
    )
    issues = list(issues_found)
    requests = tuple(
        request
        for request in build_description_requests(
            document, asset_store=asset_store, slide_context=contexts
        )
        if request.visual_id in selected_ids
    )
    evidence: list[DerivedVisualEvidence] = []
    telemetry: list[VisualEnrichmentTelemetry] = []
    visuals_by_id = {visual.visual_id: visual for visual in document.visuals}
    semaphore = asyncio.Semaphore(max_concurrency)

    async def describe(request):
        async with semaphore:
            result = await describer.describe(request)
        if result.description.visual_id != request.visual_id:
            raise ValueError("visual describer returned evidence for the wrong visual")
        return result

    if continue_on_error:
        outcomes = await asyncio.gather(
            *(describe(request) for request in requests), return_exceptions=True
        )
    else:
        outcomes = [await describe(request) for request in requests]

    for request, result in zip(requests, outcomes, strict=True):
        visual = visuals_by_id[request.visual_id]
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            issues.append(
                VisualEnrichmentIssue(
                    visual.visual_id, visual.slide_number, "failed",
                    type(result).__name__,
                )
            )
            continue
        description = result.description
        if visual.asset_hash is None:
            raise ValueError("described visual has no source asset hash")
        identity = "\0".join(
            (
                document.document_version,
                visual.visual_id,
                visual.asset_hash,
                description.model_id,
                description.prompt_version,
            )
        ).encode("utf-8")
        evidence.append(
            DerivedVisualEvidence(
                evidence_id=f"dve_{sha256(identity).hexdigest()[:20]}",
                document_id=document.document_id,
                document_version=document.document_version,
                visual_id=visual.visual_id,
                asset_hash=visual.asset_hash,
                slide_number=visual.slide_number,
                shape_id=visual.shape_id,
                model_id=description.model_id,
                prompt_version=description.prompt_version,
                literal_description=description.literal_description,
                visible_text=description.visible_text,
                entities=description.entities,
                relationships=description.relationships,
                uncertainty=description.uncertainty,
            )
        )
        telemetry.append(
            VisualEnrichmentTelemetry(
                visual_id=visual.visual_id,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
            )
        )

    return VisualEnrichmentResult(
        document_id=document.document_id,
        document_version=document.document_version,
        evidence=tuple(evidence),
        telemetry=tuple(telemetry),
        issues=tuple(issues),
    )


def _build_slide_context(
    document: ExtractedDocument, *, max_chars: int
) -> dict[int, str]:
    by_slide: dict[int, list[str]] = {}
    for block in document.blocks:
        slide_number = block.location.slide_number
        if slide_number is None:
            continue
        by_slide.setdefault(slide_number, []).append(
            f"[{block.block_type}] {block.text}"
        )
    return {
        slide: "\n".join(lines)[:max_chars]
        for slide, lines in by_slide.items()
    }
