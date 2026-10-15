"""Deterministic support checks that run before any model verification."""

from __future__ import annotations

from backend.app.chunking import Chunk
from backend.app.generation import GeneratedArtifact

from .numbers import figure_findings


def validate_deterministic_support(
    artifact: GeneratedArtifact, evidence: tuple[Chunk, ...]
) -> None:
    """Reject citations outside the supplied evidence, and figures the source does not
    contain."""
    chunks = {chunk.chunk_id: chunk for chunk in evidence}
    for block in artifact.blocks:
        for unit in block.units:
            if not unit.factual:
                continue
            for reference in unit.support:
                chunk = chunks.get(reference.chunk_id)
                if chunk is None:
                    raise ValueError(f"unit {unit.unit_id} cites an unavailable chunk")
                available_ids = {item.evidence_id for item in chunk.evidence}
                if not set(reference.evidence_ids) <= available_ids:
                    raise ValueError(f"unit {unit.unit_id} cites unavailable evidence")
    problems = []
    for finding in figure_findings(artifact, evidence):
        missing = (*finding.unsupported, *finding.image_only)
        if not missing:
            continue
        problems.append(
            f"unit {finding.unit_id} contains unsupported numeric tokens: "
            + ", ".join(missing)
            + (
                " (found only in an image description, which cannot support a "
                "number without confirming source text: "
                + ", ".join(finding.image_only) + ")"
                if finding.image_only else ""
            )
        )
    if problems:
        raise ValueError("; ".join(problems))
