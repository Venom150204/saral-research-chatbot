"""Schema-constrained Gemini claim entailment adapter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.chunking import Chunk
from backend.app.generation import GeneratedArtifact
from backend.app.prompts import VERIFICATION_PROMPT, render_prompt

from backend.app.chunking import location_label

from .models import (
    ClaimVerdict, RequirementVerdict, SuggestedAction, UserGuidance, VerificationReport,
)
from backend.app.provider_retry import call_with_retry

from .constraints import measure_constraints
from .numbers import claim_figures, figure_findings, source_values


PROMPT_VERSION = "support-and-fulfillment-v11-safety-accessibility"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
VERDICTS = (
    "supported", "partially_supported", "unsupported", "contradicted", "insufficient_evidence"
)
UNCITED_VERDICTS = ("acceptable", "needs_source", "incorrect")


class _LLMVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1)
    verdict: str
    reason: str = Field(min_length=1)
    supporting_quotes: list[str]


class _LLMSuggestedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    instruction: str = Field(min_length=1)


class _LLMUncitedVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1)
    verdict: str
    reason: str = Field(min_length=1)


class _LLMRequirementVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(min_length=1)
    met: bool
    reason: str = Field(min_length=1)


class _LLMVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[_LLMVerdict]
    uncited_verdicts: list[_LLMUncitedVerdict] = Field(default_factory=list)
    requirement_verdicts: list[_LLMRequirementVerdict] = Field(default_factory=list)
    fulfills_request: bool = True
    quality_reason: str = Field(
        default="The artifact fulfills the requested deliverable.", min_length=1
    )
    quality_issues: list[str] = Field(default_factory=list)
    guidance_summary: str = ""
    guidance_observations: list[str] = Field(default_factory=list)
    guidance_attempts: list[str] = Field(default_factory=list)
    suggested_actions: list[_LLMSuggestedAction] = Field(default_factory=list)
    can_retry_automatically: bool = False


class VerificationTransport(Protocol):
    async def generate(self, *, model_id: str, prompt: str, response_schema: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GoogleVerificationTransport:
    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def generate(self, *, model_id, prompt, response_schema):
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=45_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        async with client.aio as async_client:
            response = await async_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=dict(response_schema),
                    temperature=0,
                    max_output_tokens=_verification_token_budget(response_schema),
                ),
            )
        payload = response.parsed or json.loads(response.text or "{}")
        if not isinstance(payload, Mapping):
            raise ValueError("verifier returned no structured payload")
        return payload


@dataclass(slots=True)
class GeminiSupportVerifier:
    transport: VerificationTransport
    model_id: str = DEFAULT_MODEL

    async def verify(self, artifact: GeneratedArtifact, evidence: tuple[Chunk, ...]) -> VerificationReport:
        factual = [unit for block in artifact.blocks for unit in block.units if unit.factual]
        chunks = {chunk.chunk_id: chunk for chunk in evidence}
        uncited_figures = {
            finding.unit_id: list(finding.cited_elsewhere)
            for finding in figure_findings(artifact, evidence)
            if finding.cited_elsewhere
        }
        items = []
        allowed_quote_text: dict[str, str] = {}
        for unit in factual:
            cited = []
            for support in unit.support:
                if support.chunk_id not in chunks:
                    raise ValueError("semantic verifier received unavailable cited evidence")
                cited_chunk = chunks[support.chunk_id]
                entry = {"chunk_id": support.chunk_id, "text": cited_chunk.text}
                location = location_label(cited_chunk.evidence)
                if location is not None:
                    entry["location"] = location
                if cited_chunk.evidence_kind == "derived":
                    entry["source"] = "image_description"
                cited.append(entry)
            item = {"unit_id": unit.unit_id, "claim": unit.text, "cited_evidence": cited}
            if unit.unit_id in uncited_figures:
                item["figures_not_in_cited_evidence"] = uncited_figures[unit.unit_id]
            items.append(item)
            allowed_quote_text[unit.unit_id] = "\n".join(item["text"] for item in cited)
        in_source = source_values(
            chunk.text for chunk in evidence if chunk.supports_quantitative_claims
        )
        unit_count = sum(len(block.units) for block in artifact.blocks)
        uncited = [unit for block in artifact.blocks for unit in block.units if not unit.factual]
        uncited_items = []
        for unit in uncited:
            item = {"unit_id": unit.unit_id, "basis": unit.basis, "text": unit.text}
            figures = [
                written for written, value in claim_figures(unit.text, item_count=unit_count)
                if value in in_source
            ]
            if figures:
                item["figures_found_in_the_source"] = figures
            uncited_items.append(item)
        output = artifact.request.output
        requirements = {f"R{index}": text for index, text in enumerate(output.requirements, start=1)}
        schema = _schema(
            tuple(unit.unit_id for unit in factual), tuple(unit.unit_id for unit in uncited),
            tuple(requirements),
        )
        artifact_view = {
            "user_request": output.original_request or output.format,
            "output_specification": output.format,
            "requirements": [
                {"requirement_id": key, "requirement": text} for key, text in requirements.items()
            ],
            "server_measurements": [
                {"required": item.required, "measured": item.measured, "met": item.met}
                for item in measure_constraints(artifact)
            ],
            "blocks": [
                {
                    "title": block.title,
                    "block_type": block.block_type,
                    "rendered_text": " ".join(unit.text for unit in block.units),
                    "units": [
                        {"unit_id": unit.unit_id, "role": unit.role, "text": unit.text}
                        for unit in block.units
                    ],
                }
                for block in artifact.blocks
            ],
        }
        prompt = render_prompt(
            VERIFICATION_PROMPT,
            prompt_name="verification",
            FACTUAL_ITEMS=json.dumps(items, ensure_ascii=False),
            UNCITED_ITEMS=json.dumps(uncited_items, ensure_ascii=False),
            ARTIFACT_TO_REVIEW=json.dumps(artifact_view, ensure_ascii=False),
        )
        started = time.perf_counter()
        expected_ids = tuple(unit.unit_id for unit in factual)
        uncited_ids = tuple(unit.unit_id for unit in uncited)
        payload = await call_with_retry(lambda: self.transport.generate(
            model_id=self.model_id, prompt=prompt, response_schema=schema
        ))
        try:
            validated_payload = _LLMVerificationResponse.model_validate(payload).model_dump()
            verdicts = _parse(validated_payload, expected_ids, allowed_quote_text)
            uncited_verdicts = _parse_uncited(validated_payload, uncited_ids)
            requirement_verdicts = _parse_requirements(validated_payload, requirements)
        except (ValidationError, ValueError, TypeError, KeyError) as first_error:
            correction = (
                prompt
                + "\n\n<STRUCTURED_OUTPUT_CORRECTION>\n"
                + "Your previous response failed server validation: "
                + str(first_error)[:500]
                + ". Return a complete replacement JSON object matching the schema exactly. "
                  "Include every required unit_id exactly once.\n"
                  "</STRUCTURED_OUTPUT_CORRECTION>"
            )
            payload = await call_with_retry(lambda: self.transport.generate(
                model_id=self.model_id, prompt=correction, response_schema=schema
            ))
            validated_payload = _LLMVerificationResponse.model_validate(payload).model_dump()
            verdicts = _parse(validated_payload, expected_ids, allowed_quote_text)
            uncited_verdicts = _parse_uncited(validated_payload, uncited_ids)
            requirement_verdicts = _parse_requirements(validated_payload, requirements)
        issues = tuple(item.strip() for item in validated_payload["quality_issues"] if item.strip())
        issues = tuple(
            f"Not met: {item.requirement} ({item.reason})"
            for item in requirement_verdicts if not item.met
        ) + issues
        fulfills = validated_payload["fulfills_request"] and not issues
        return VerificationReport(
            verdicts, self.model_id, PROMPT_VERSION,
            (time.perf_counter() - started) * 1000,
            fulfills, validated_payload["quality_reason"].strip(), issues,
            UserGuidance(
                summary=validated_payload["guidance_summary"].strip(),
                observations=tuple(
                    item.strip() for item in validated_payload["guidance_observations"]
                    if item.strip()
                ),
                attempts=tuple(
                    item.strip() for item in validated_payload["guidance_attempts"]
                    if item.strip()
                ),
                suggested_actions=tuple(
                    SuggestedAction(item["label"].strip(), item["instruction"].strip())
                    for item in validated_payload["suggested_actions"]
                    if item.get("label", "").strip() and item.get("instruction", "").strip()
                ),
                can_retry_automatically=validated_payload["can_retry_automatically"],
            ),
            uncited_verdicts,
            requirement_verdicts,
        )


def _parse_requirements(payload, requirements: dict[str, str]) -> tuple[RequirementVerdict, ...]:
    raw = payload.get("requirement_verdicts") or []
    by_id = {item["requirement_id"]: item for item in raw}
    if len(raw) != len(requirements) or set(by_id) != set(requirements):
        raise ValueError("verifier must judge every requirement exactly once")
    return tuple(
        RequirementVerdict(text, bool(by_id[key]["met"]), by_id[key]["reason"].strip())
        for key, text in requirements.items()
    )


def _verification_token_budget(response_schema: Mapping[str, Any]) -> int:
    """Allow enough JSON for one verdict per unit without unbounded responses."""
    properties = response_schema.get("properties", {})
    count = sum(
        int(properties.get(name, {}).get("maxItems", 0) or 0)
        for name in ("verdicts", "uncited_verdicts", "requirement_verdicts")
    )
    return min(12_000, max(3_200, 1_500 + count * 220))


def _schema(
    unit_ids: tuple[str, ...], uncited_ids: tuple[str, ...] = (), requirement_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    uncited_id = {"type": "string", "enum": list(uncited_ids)} if uncited_ids else {"type": "string"}
    requirement_id = (
        {"type": "string", "enum": list(requirement_ids)} if requirement_ids else {"type": "string"}
    )
    return {
        "type": "object",
        "properties": {
        "uncited_verdicts": {"type": "array", "minItems": len(uncited_ids), "maxItems": len(uncited_ids), "items": {
            "type": "object", "properties": {
                "unit_id": uncited_id,
                "verdict": {"type": "string", "enum": list(UNCITED_VERDICTS)},
                "reason": {"type": "string"},
            }, "required": ["unit_id", "verdict", "reason"], "additionalProperties": False,
        }},
        "verdicts": {"type": "array", "minItems": len(unit_ids), "maxItems": len(unit_ids), "items": {
            "type": "object", "properties": {
                "unit_id": {"type": "string", "enum": list(unit_ids)},
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "reason": {"type": "string"},
                "supporting_quotes": {
                    "type": "array", "maxItems": 0, "items": {"type": "string"}
                },
            }, "required": ["unit_id", "verdict", "reason", "supporting_quotes"], "additionalProperties": False,
        }},
        "requirement_verdicts": {"type": "array", "minItems": len(requirement_ids), "maxItems": len(requirement_ids), "items": {
            "type": "object", "properties": {
                "requirement_id": requirement_id,
                "met": {"type": "boolean"},
                "reason": {"type": "string"},
            }, "required": ["requirement_id", "met", "reason"], "additionalProperties": False,
        }},
        "fulfills_request": {"type": "boolean"},
        "quality_reason": {"type": "string"},
        "quality_issues": {"type": "array", "items": {"type": "string"}},
        "guidance_summary": {"type": "string"},
        "guidance_observations": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
        "guidance_attempts": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "suggested_actions": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "properties": {
                "label": {"type": "string"}, "instruction": {"type": "string"},
            }, "required": ["label", "instruction"], "additionalProperties": False,
        }},
        "can_retry_automatically": {"type": "boolean"},
        }, "required": ["verdicts", "uncited_verdicts", "requirement_verdicts", "fulfills_request", "quality_reason", "quality_issues", "guidance_summary", "guidance_observations", "guidance_attempts", "suggested_actions", "can_retry_automatically"], "additionalProperties": False,
    }


def _parse_uncited(payload, expected_ids) -> tuple[ClaimVerdict, ...]:
    raw = payload.get("uncited_verdicts") or []
    if len(raw) != len(expected_ids):
        raise ValueError("verifier must review every uncited unit exactly once")
    verdicts = []
    for item in raw:
        if item.get("unit_id") not in expected_ids or item.get("verdict") not in UNCITED_VERDICTS:
            raise ValueError("verifier returned an invalid review of uncited text")
        verdicts.append(ClaimVerdict(item["unit_id"], item["verdict"], item["reason"].strip()))
    if {item.unit_id for item in verdicts} != set(expected_ids):
        raise ValueError("verifier returned duplicate or missing uncited unit IDs")
    return tuple(verdicts)


def _parse(payload, expected_ids, allowed_quote_text) -> tuple[ClaimVerdict, ...]:
    raw = payload.get("verdicts")
    if not isinstance(raw, list) or len(raw) != len(expected_ids):
        raise ValueError("verifier must return every factual unit exactly once")
    verdicts = []
    for item in raw:
        if not isinstance(item, Mapping) or item.get("unit_id") not in expected_ids or item.get("verdict") not in VERDICTS:
            raise ValueError("verifier returned an invalid verdict")
        quotes = item.get("supporting_quotes")
        reason = item.get("reason")
        if not isinstance(quotes, list) or any(not isinstance(q, str) for q in quotes) or not isinstance(reason, str) or not reason.strip():
            raise ValueError("verifier reason and quotes are invalid")
        searchable = allowed_quote_text[item["unit_id"]]
        if any(quote not in searchable for quote in quotes):
            raise ValueError("verifier quote is not present in supplied evidence")
        verdicts.append(ClaimVerdict(item["unit_id"], item["verdict"], reason.strip(), tuple(quotes)))
    if {item.unit_id for item in verdicts} != set(expected_ids):
        raise ValueError("verifier returned duplicate or missing unit IDs")
    return tuple(verdicts)
