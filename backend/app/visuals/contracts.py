"""Typed boundary between extracted visuals and a future VLM provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from backend.app.domain import ExtractedDocument
from backend.app.storage import AssetStore, BinaryAsset


@dataclass(frozen=True, slots=True)
class VisualDescriptionRequest:
    """Internal request; image bytes are encoded only inside a provider adapter."""

    visual_id: str
    slide_number: int
    shape_id: int
    asset: BinaryAsset
    slide_context: str


@dataclass(frozen=True, slots=True)
class VisualDescription:
    """Serializable, explicitly model-derived description of one visual."""

    visual_id: str
    model_id: str
    prompt_version: str
    literal_description: str
    visible_text: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    uncertainty: str | None = None
    derived: bool = True

    def __post_init__(self) -> None:
        if not self.literal_description.strip():
            raise ValueError("visual description cannot be empty")
        if not self.derived:
            raise ValueError("VLM descriptions must remain marked as derived")


@dataclass(frozen=True, slots=True)
class VisualDescriptionResult:
    """One description plus provider telemetry used for evaluation."""

    description: VisualDescription
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")


class VisionDescriber(Protocol):
    """Local and hosted VLM adapters implement the same async contract."""

    async def describe(
        self, request: VisualDescriptionRequest
    ) -> VisualDescriptionResult: ...


def build_description_requests(
    document: ExtractedDocument,
    *,
    asset_store: AssetStore,
    slide_context: Mapping[int, str],
) -> tuple[VisualDescriptionRequest, ...]:
    """Select persisted pictures without guessing whether they are decorative."""

    requests: list[VisualDescriptionRequest] = []
    for visual in document.visuals:
        if visual.visual_type != "picture" or visual.asset_hash is None:
            continue
        requests.append(
            VisualDescriptionRequest(
                visual_id=visual.visual_id,
                slide_number=visual.slide_number,
                shape_id=visual.shape_id,
                asset=asset_store.get(visual.asset_hash),
                slide_context=slide_context.get(visual.slide_number, ""),
            )
        )
    return tuple(requests)
