"""Gemini edit-proposal tool; it never persists mutations itself."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.generation import GeneratedArtifact, GroundedText, SupportReference
from backend.app.generation.providers.gemini import GenerationTransport
from backend.app.prompts import EDITING_PROMPT, render_prompt
from backend.app.provider_retry import call_with_retry

from .models import EditResult, UnitDelta


PROMPT_VERSION = "scoped-edit-proposal-v4-words-not-directions"
DEFAULT_MODEL = "gemini-3.7-flash"


class _Support(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class _Replacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    basis: Literal["source", "background", "framing"]
    support: list[_Support]


class _EditProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1)
    replacements: list[_Replacement] = Field(min_length=1)


class GeminiArtifactEditor:
    def __init__(self, transport: GenerationTransport, *, model_id: str = DEFAULT_MODEL) -> None:
        self._transport = transport
        self._model_id = model_id

    async def edit(
        self, *, artifact: GeneratedArtifact, target_unit_ids: tuple[str, ...],
        instruction: str, evidence: tuple,
    ) -> EditResult:
        allowed_evidence = {
            chunk.chunk_id: {item.evidence_id for item in chunk.evidence}
            for chunk in evidence
        }
        units = {unit.unit_id: unit for block in artifact.blocks for unit in block.units}
        if not set(target_unit_ids) <= units.keys():
            raise ValueError("edit tool received unknown target unit IDs")
        prompt = _prompt(artifact, target_unit_ids, instruction, evidence)
        schema = _schema(target_unit_ids, allowed_evidence)
        response = await call_with_retry(lambda: self._transport.generate(
            model_id=self._model_id, prompt=prompt, response_schema=schema,
        ))
        proposal = _EditProposal.model_validate(response.payload)
        replacements: dict[str, GroundedText] = {}
        for item in proposal.replacements:
            if item.unit_id not in target_unit_ids or item.unit_id in replacements:
                raise ValueError("edit proposal contains duplicate or out-of-scope units")
            support = tuple(
                SupportReference(ref.chunk_id, tuple(ref.evidence_ids))
                for ref in item.support
            )
            for ref in support:
                if ref.chunk_id not in allowed_evidence or not set(ref.evidence_ids) <= allowed_evidence[ref.chunk_id]:
                    raise ValueError("edit proposal cited evidence outside the active document")
            original = units[item.unit_id]
            source = item.basis == "source"
            replacements[item.unit_id] = replace(
                original, text=item.text.strip(), factual=source, basis=item.basis,
                support=support if source else (),
            )
        blocks = tuple(
            replace(
                block,
                units=tuple(replacements.get(unit.unit_id, unit) for unit in block.units),
            )
            for block in artifact.blocks
        )
        edited = replace(
            artifact, version=artifact.version + 1, blocks=blocks,
            model_id=self._model_id, prompt_version=PROMPT_VERSION,
        )
        deltas = tuple(
            UnitDelta(unit_id, units[unit_id].text, replacement.text)
            for unit_id, replacement in replacements.items()
            if replacement != units[unit_id]
        )
        return EditResult(
            edited, deltas, proposal.reason.strip(), self._model_id, PROMPT_VERSION
        )


def _prompt(artifact, target_unit_ids, instruction, evidence) -> str:
    target_units = [
        asdict(unit)
        for block in artifact.blocks for unit in block.units
        if unit.unit_id in target_unit_ids
    ]
    source = [
        {
            "chunk_id": chunk.chunk_id,
            "evidence_ids": [item.evidence_id for item in chunk.evidence],
            "text": chunk.text,
        }
        for chunk in evidence
    ]
    return render_prompt(
        EDITING_PROMPT,
        prompt_name="editing",
        INSTRUCTION=instruction,
        TARGET_UNITS=json.dumps(target_units, ensure_ascii=False),
        EVIDENCE=json.dumps(source, ensure_ascii=False),
    )


def _schema(target_unit_ids, allowed_evidence) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "replacements": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "unit_id": {"type": "string", "enum": list(target_unit_ids)},
                        "text": {"type": "string"},
                        "basis": {"type": "string", "enum": ["source", "background", "framing"]},
                        "support": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "chunk_id": {"type": "string", "enum": list(allowed_evidence)},
                                "evidence_ids": {"type": "array", "minItems": 1, "items": {
                                    "type": "string",
                                    "enum": sorted({item for ids in allowed_evidence.values() for item in ids}),
                                }},
                            },
                            "required": ["chunk_id", "evidence_ids"],
                            "additionalProperties": False,
                        }},
                    },
                    "required": ["unit_id", "text", "basis", "support"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reason", "replacements"],
        "additionalProperties": False,
    }
