"""Scope-safe artifact editing and verification."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from backend.app.generation import GeneratedArtifact
from backend.app.verification import (
    complete_numeric_citations, validate_deterministic_support,
)

from .models import EditResult


class ArtifactEditor(Protocol):
    async def edit(
        self, *, artifact: GeneratedArtifact, target_unit_ids: tuple[str, ...],
        instruction: str, evidence: tuple,
    ) -> EditResult: ...


class EditRejectedError(ValueError):
    """An edit that could not be applied safely. `user_message` says why in plain
    words; the previous version stays current."""

    def __init__(self, user_message: str, *, details: tuple[str, ...] = ()) -> None:
        super().__init__(user_message + (" Details: " + "; ".join(details) if details else ""))
        self.user_message = user_message
        self.details = details


def artifact_outline(artifact: GeneratedArtifact) -> list[dict]:
    """A compact map of the draft, numbered in reading order, so a request such as
    "make #2 simpler" or "shorten the opening" can be resolved to a real ID."""
    return [
        {
            "number": number,
            "block_id": block.block_id,
            "title": block.title or block.block_type,
            "units": [
                {"unit_id": unit.unit_id, "role": unit.role,
                 "starts": " ".join(unit.text.split()[:12])}
                for unit in block.units
            ],
        }
        for number, block in enumerate(artifact.blocks, start=1)
    ]


def _only_units(artifact: GeneratedArtifact, unit_ids: set[str]) -> GeneratedArtifact:
    blocks = tuple(
        replace(block, units=tuple(unit for unit in block.units if unit.unit_id in unit_ids))
        for block in artifact.blocks
        if any(unit.unit_id in unit_ids for unit in block.units)
    )
    output = replace(artifact.request.output, requirements=())
    return replace(artifact, blocks=blocks, request=replace(artifact.request, output=output))


class VerifiedEditService:
    def __init__(self, *, editor: ArtifactEditor, verifier) -> None:
        self._editor = editor
        self._verifier = verifier

    async def edit(
        self, *, artifact: GeneratedArtifact, target_id: str,
        instruction: str, evidence: tuple,
    ) -> EditResult:
        try:
            target_units = resolve_artifact_target(artifact, target_id)
        except ValueError as error:
            raise EditRejectedError(
                "I couldn't tell which part of the draft you meant. Name it as it "
                "appears, for example \"the second slide\" or \"the opening\".",
                details=(str(error),),
            ) from error
        result = await self._editor.edit(
            artifact=artifact, target_unit_ids=target_units,
            instruction=instruction, evidence=evidence,
        )
        try:
            validate_edit_result(artifact, result, set(target_units))
        except ValueError as error:
            raise EditRejectedError(
                "The edit could not be applied safely, so the previous version is kept.",
                details=(str(error),),
            ) from error
        completed, _ = complete_numeric_citations(
            result.artifact, evidence,
            unit_ids={delta.unit_id for delta in result.deltas},
        )
        result = replace(result, artifact=completed)
        try:
            validate_deterministic_support(result.artifact, evidence)
        except ValueError as error:
            raise EditRejectedError(
                "The edited text used figures or citations the deck does not support, "
                "so the previous version is kept.",
                details=(str(error),),
            ) from error
        changed = {delta.unit_id for delta in result.deltas}
        report = await self._verifier.verify(_only_units(result.artifact, changed), evidence)
        if not report.accepted:
            raise EditRejectedError(
                "The edited text said things the deck does not support, so the "
                "previous version is kept.",
                details=tuple(f"{item.unit_id}: {item.reason}" for item in report.rejected),
            )
        return result


def resolve_artifact_target(artifact: GeneratedArtifact, target_id: str) -> tuple[str, ...]:
    target = _normalize(target_id)
    all_units = tuple(unit for block in artifact.blocks for unit in block.units)
    if target in {"artifact", "current-artifact", _normalize(artifact.artifact_id)}:
        return tuple(unit.unit_id for unit in all_units)
    for unit in all_units:
        if _normalize(unit.unit_id) == target:
            return (unit.unit_id,)
    for block in artifact.blocks:
        if _normalize(block.block_id) == target:
            return tuple(unit.unit_id for unit in block.units)
    raise ValueError(f"artifact target {target_id!r} is ambiguous or does not exist")


def validate_edit_result(
    before: GeneratedArtifact, result: EditResult, allowed: set[str]
) -> None:
    after = result.artifact
    if after.artifact_id != before.artifact_id or after.version != before.version + 1:
        raise ValueError("edit must preserve artifact identity and increment one version")
    if after.request != before.request:
        raise ValueError("edit cannot change the artifact's source or output specification")
    before_blocks = [(b.block_id, b.block_type, b.title, b.attributes) for b in before.blocks]
    after_blocks = [(b.block_id, b.block_type, b.title, b.attributes) for b in after.blocks]
    if before_blocks != after_blocks:
        raise ValueError("scoped text edit cannot change artifact block structure")
    before_units = {unit.unit_id: unit for block in before.blocks for unit in block.units}
    after_units = {unit.unit_id: unit for block in after.blocks for unit in block.units}
    if before_units.keys() != after_units.keys():
        raise ValueError("scoped text edit must preserve all stable unit IDs")
    changed = {unit_id for unit_id in before_units if before_units[unit_id] != after_units[unit_id]}
    if not changed:
        raise ValueError("edit produced no change")
    if not changed <= allowed:
        raise ValueError("edit changed content outside the resolved target")
    delta_ids = {delta.unit_id for delta in result.deltas}
    if delta_ids != changed or len(delta_ids) != len(result.deltas):
        raise ValueError("edit deltas must describe every changed unit exactly once")
    for delta in result.deltas:
        if delta.old_text != before_units[delta.unit_id].text:
            raise ValueError("edit delta old text does not match the base revision")
        if delta.new_text != after_units[delta.unit_id].text:
            raise ValueError("edit delta new text does not match the edited revision")
    if not result.reason.strip():
        raise ValueError("edit requires a user-visible change reason")


def _normalize(value: str) -> str:
    return "-".join(value.strip().casefold().replace("_", "-").split())
