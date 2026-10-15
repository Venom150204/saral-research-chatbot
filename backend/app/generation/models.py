"""Provider-neutral contracts for open-ended, evidence-grounded artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

UNIT_BASES = ("source", "background", "framing")

CONSTRAINT_KINDS = ("item_count", "max_characters", "max_words", "duration_seconds")
CONSTRAINT_SCOPES = ("whole", "each_item")
CONSTRAINT_BASES = ("stated", "implied")


@dataclass(frozen=True, slots=True)
class DeliverableConstraint:
    """One limit on the deliverable that the server can measure."""

    kind: str
    value: int
    quote: str
    scope: str = "whole"
    basis: str = "stated"

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError("unsupported constraint kind")
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 1:
            raise ValueError("a constraint value must be a positive whole number")
        if self.scope not in CONSTRAINT_SCOPES or self.basis not in CONSTRAINT_BASES:
            raise ValueError("unsupported constraint scope or basis")
        if self.kind == "item_count" and self.scope != "whole":
            raise ValueError("an item count applies to the whole deliverable")
        if not self.quote.strip():
            raise ValueError("a constraint must quote the user's words")


@dataclass(frozen=True, slots=True)
class OutputSpecification:
    """User-owned output requirements without a closed format/style taxonomy."""

    format: str
    audience: str | None = None
    purpose: str | None = None
    style_guidance: str | None = None
    requested_count: int | None = None
    target_duration_seconds: int | None = None
    preserve_math: bool = True
    requirements: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    original_request: str | None = None
    constraints: tuple[DeliverableConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.format.strip():
            raise ValueError("an open-ended output format is required")
        if self.audience is not None and not self.audience.strip():
            raise ValueError("audience cannot be blank")
        if self.requested_count is not None and self.requested_count < 1:
            raise ValueError("requested_count must be positive")
        if self.target_duration_seconds is not None and self.target_duration_seconds < 1:
            raise ValueError("target_duration_seconds must be positive")
        if any(not item.strip() for item in self.requirements):
            raise ValueError("output requirements cannot be blank")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    document_id: str
    document_version: str
    output: OutputSpecification
    revision_feedback: tuple[str, ...] = ()
    previous_draft: str | None = None
    current_draft: str | None = None
    conversation: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.document_version.strip():
            raise ValueError("document identity and version are required")


@dataclass(frozen=True, slots=True)
class SupportReference:
    """A claimed support link that the server must resolve and validate."""

    chunk_id: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id is required")
        if not self.evidence_ids or any(not value.strip() for value in self.evidence_ids):
            raise ValueError("at least one evidence ID is required")


@dataclass(frozen=True, slots=True)
class GroundedText:
    """One stable output unit for citation checks and targeted edits."""

    unit_id: str
    role: str
    text: str
    factual: bool
    support: tuple[SupportReference, ...] = ()
    contains_math: bool = False
    delivery: str | None = None
    basis: str | None = None

    def __post_init__(self) -> None:
        if not self.unit_id.strip() or not self.role.strip() or not self.text.strip():
            raise ValueError("grounded text requires an ID, open role, and text")
        if self.factual and not self.support:
            raise ValueError("factual output units require source support")
        basis = self.basis or ("source" if self.factual else "framing")
        if basis not in UNIT_BASES:
            raise ValueError("unsupported unit basis")
        if self.factual != (basis == "source"):
            raise ValueError("a unit is factual exactly when its basis is source")
        object.__setattr__(self, "basis", basis)


@dataclass(frozen=True, slots=True)
class ArtifactBlock:
    """A generic ordered block; block type and unit roles are model/user defined."""

    block_id: str
    block_type: str
    units: tuple[GroundedText, ...]
    title: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id.strip() or not self.block_type.strip():
            raise ValueError("artifact blocks require an ID and open block type")
        if not self.units:
            raise ValueError("artifact blocks cannot be empty")


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    artifact_id: str
    version: int
    request: GenerationRequest
    blocks: tuple[ArtifactBlock, ...]
    model_id: str
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id is required")
        if self.version < 1:
            raise ValueError("artifact version must be positive")
        if not self.blocks:
            raise ValueError("generated artifacts require content blocks")
        if not self.model_id.strip() or not self.prompt_version.strip():
            raise ValueError("model and prompt versions are required")

    @property
    def spoken_word_count(self) -> int:
        return sum(
            len(unit.text.split())
            for block in self.blocks
            for unit in block.units
            if unit.delivery and unit.delivery.casefold() == "spoken"
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    artifact: GeneratedArtifact
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def artifact_from_dict(payload: Mapping[str, Any]) -> GeneratedArtifact:
    """Restore a persisted artifact through the same domain invariants."""
    request_payload = payload["request"]
    output_payload = request_payload["output"]
    request = GenerationRequest(
        document_id=request_payload["document_id"],
        document_version=request_payload["document_version"],
        output=OutputSpecification(
            format=output_payload["format"],
            audience=output_payload.get("audience"),
            purpose=output_payload.get("purpose"),
            style_guidance=output_payload.get("style_guidance"),
            requested_count=output_payload.get("requested_count"),
            target_duration_seconds=output_payload.get("target_duration_seconds"),
            preserve_math=output_payload.get("preserve_math", True),
            requirements=tuple(output_payload.get("requirements", ())),
            attributes=dict(output_payload.get("attributes", {})),
            original_request=output_payload.get("original_request"),
            constraints=tuple(
                DeliverableConstraint(
                    kind=item["kind"], value=item["value"], quote=item["quote"],
                    scope=item.get("scope", "whole"), basis=item.get("basis", "stated"),
                )
                for item in output_payload.get("constraints", ())
            ),
        ),
        revision_feedback=tuple(request_payload.get("revision_feedback", ())),
        previous_draft=request_payload.get("previous_draft"),
        current_draft=request_payload.get("current_draft"),
        conversation=request_payload.get("conversation"),
    )
    blocks = tuple(
        ArtifactBlock(
            block_id=block["block_id"],
            block_type=block["block_type"],
            title=block.get("title"),
            attributes=dict(block.get("attributes", {})),
            units=tuple(
                GroundedText(
                    unit_id=unit["unit_id"], role=unit["role"], text=unit["text"],
                    factual=unit["factual"], contains_math=unit.get("contains_math", False),
                    delivery=unit.get("delivery"), basis=unit.get("basis"),
                    support=tuple(
                        SupportReference(item["chunk_id"], tuple(item["evidence_ids"]))
                        for item in unit.get("support", ())
                    ),
                )
                for unit in block["units"]
            ),
        )
        for block in payload["blocks"]
    )
    return GeneratedArtifact(
        artifact_id=payload["artifact_id"], version=payload["version"],
        request=request, blocks=blocks, model_id=payload["model_id"],
        prompt_version=payload["prompt_version"],
    )
