"""Provider-neutral conversational editing contracts."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.generation import GeneratedArtifact


@dataclass(frozen=True, slots=True)
class UnitDelta:
    unit_id: str
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class EditResult:
    artifact: GeneratedArtifact
    deltas: tuple[UnitDelta, ...]
    reason: str
    model_id: str
    prompt_version: str

