"""Gemini adapter for grounded, audience-adaptive communication artifacts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from backend.app.chunking import Chunk, location_label
from backend.app.generation.models import (
    ArtifactBlock,
    GeneratedArtifact,
    GenerationRequest,
    GenerationResult,
    GroundedText,
    SupportReference,
)
from backend.app.prompts import GENERATION_PROMPT, render_prompt
from backend.app.provider_retry import call_with_retry

PROMPT_VERSION = "grounded-authoring-v15-safe-accessible"
DEFAULT_MODEL = "gemini-3.7-flash"

OUTPUT_TOKENS_PER_WORD = 6.5
DEFAULT_OUTPUT_TOKENS = 8_000
TYPICAL_OUTPUT_TOKENS = 2_500
SIZE_MARGIN = 1.5
REASONING_RESERVE_TOKENS = 4_000
BASE_TIMEOUT_SECONDS = 30.0
WRITING_TOKENS_PER_SECOND = 40.0
MIN_TIMEOUT_SECONDS = 45.0
MAX_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """What the provider reports about a model. Unknown values stay None/False."""

    output_token_limit: int | None = None
    thinks: bool = False


@dataclass(frozen=True, slots=True)
class WritingBudget:
    max_output_tokens: int
    timeout_seconds: float


def _timeout_for(expected_tokens: float) -> float:
    seconds = BASE_TIMEOUT_SECONDS + expected_tokens / WRITING_TOKENS_PER_SECOND
    return min(MAX_TIMEOUT_SECONDS, max(MIN_TIMEOUT_SECONDS, seconds))


def writing_budget(request: GenerationRequest, profile: ModelProfile) -> WritingBudget:
    """Room and time for one writing call, sized from what the user asked for."""
    from backend.app.verification.constraints import requested_words

    words = requested_words(request.output.constraints)
    reserve = REASONING_RESERVE_TOKENS if profile.thinks else 0
    room = DEFAULT_OUTPUT_TOKENS if words is None else max(
        DEFAULT_OUTPUT_TOKENS, math.ceil(words * OUTPUT_TOKENS_PER_WORD * SIZE_MARGIN)
    )
    room += reserve
    if profile.output_token_limit:
        room = min(room, profile.output_token_limit)
    expected = (words * OUTPUT_TOKENS_PER_WORD if words else TYPICAL_OUTPUT_TOKENS) + reserve
    return WritingBudget(room, _timeout_for(expected))


class OutputTruncatedError(ValueError):
    """The provider stopped because the answer reached the output limit."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"the answer was cut off at the {limit}-token output limit")
        self.limit = limit


class RequestTooLongError(ValueError):
    """The requested deliverable does not fit in one draft, even with more room."""

    retryable = False

    def __init__(self, words_that_fit: int) -> None:
        self.user_message = (
            "This request needs a longer answer than one draft can hold (about "
            f"{words_that_fit} words fit in one draft). Ask for fewer items or a shorter "
            "length, or split it: for example the first half now and the rest next."
        )
        super().__init__(self.user_message)


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return None
    return getattr(reason, "name", None) or str(reason)


class _LLMSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)


class _LLMUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    text: str = Field(min_length=1)
    basis: Literal["source", "background", "framing"]
    support: list[_LLMSupport]
    contains_math: bool
    delivery: str | None


class _LLMBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str = Field(min_length=1)
    block_type: str = Field(min_length=1)
    title: str | None
    units: list[_LLMUnit] = Field(min_length=1)


class _LLMGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[_LLMBlock] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GenerationTransport(Protocol):
    async def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
        timeout_seconds: float = MIN_TIMEOUT_SECONDS,
    ) -> ProviderResponse: ...


class GoogleGenAIGenerationTransport:
    _profiles: dict[str, ModelProfile] = {}

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def model_profile(self, model_id: str) -> ModelProfile:
        """The model's maximum output and whether it reasons, as Google reports."""
        if model_id in self._profiles:
            return self._profiles[model_id]
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=10_000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        try:
            async with client.aio as async_client:
                model = await async_client.models.get(model=model_id)
        except Exception:
            return ModelProfile()
        profile = ModelProfile(model.output_token_limit, bool(model.thinking))
        self._profiles[model_id] = profile
        return profile

    async def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
        timeout_seconds: float = MIN_TIMEOUT_SECONDS,
    ) -> ProviderResponse:
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
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
                    temperature=0.45,
                    max_output_tokens=max_output_tokens,
                ),
            )
        if _finish_reason(response) == "MAX_TOKENS":
            raise OutputTruncatedError(max_output_tokens)
        payload = response.parsed
        if payload is None and response.text:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError as error:
                finish_reason = None
                if getattr(response, "candidates", None):
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)
                raise ValueError(
                    "Gemini returned incomplete structured generation output"
                    + (f" (finish_reason={finish_reason})" if finish_reason else "")
                ) from error
        if not isinstance(payload, Mapping):
            raise ValueError("Gemini returned no structured generation payload")
        usage = response.usage_metadata
        return ProviderResponse(
            payload=payload,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )


class GeminiGroundedGenerator:
    def __init__(
        self,
        transport: GenerationTransport,
        *,
        model_id: str = DEFAULT_MODEL,
    ) -> None:
        self._transport = transport
        self._model_id = model_id

    async def generate(
        self,
        request: GenerationRequest,
        evidence: tuple[Chunk, ...],
    ) -> GenerationResult:
        _validate_evidence_scope(request, evidence)
        schema = _response_schema(evidence)
        prompt = _build_prompt(request, evidence)
        started = time.perf_counter()
        profile = await self._profile()
        budget = writing_budget(request, profile)
        try:
            response = await self._generate_with_retry(prompt=prompt, schema=schema, budget=budget)
        except OutputTruncatedError:
            room = budget.max_output_tokens * 2
            if profile.output_token_limit:
                room = min(room, profile.output_token_limit)
            if room <= budget.max_output_tokens:
                raise RequestTooLongError(_words_that_fit(budget.max_output_tokens, profile)) from None
            larger = WritingBudget(room, min(MAX_TIMEOUT_SECONDS, budget.timeout_seconds * 2))
            try:
                response = await self._generate_with_retry(prompt=prompt, schema=schema, budget=larger)
            except OutputTruncatedError as error:
                raise RequestTooLongError(_words_that_fit(room, profile)) from error
        validated_payload = _LLMGenerationResponse.model_validate(response.payload).model_dump()
        blocks = _parse_blocks(validated_payload, evidence)
        identity = json.dumps(
            {
                "request": asdict(request),
                "model": self._model_id,
                "prompt_version": PROMPT_VERSION,
                "payload": response.payload,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        artifact = GeneratedArtifact(
            artifact_id=f"art_{sha256(identity).hexdigest()[:20]}",
            version=1,
            request=request,
            blocks=blocks,
            model_id=self._model_id,
            prompt_version=PROMPT_VERSION,
        )
        return GenerationResult(
            artifact=artifact,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )

    async def _profile(self) -> ModelProfile:
        lookup = getattr(self._transport, "model_profile", None)
        return await lookup(self._model_id) if lookup else ModelProfile()

    async def _generate_with_retry(
        self, *, prompt: str, schema: Mapping[str, Any], budget: WritingBudget,
    ) -> ProviderResponse:
        """Retry a transient provider failure; never retry invalid model output."""
        return await call_with_retry(lambda: self._transport.generate(
            model_id=self._model_id, prompt=prompt, response_schema=schema,
            max_output_tokens=budget.max_output_tokens,
            timeout_seconds=budget.timeout_seconds,
        ))


def _words_that_fit(room: int, profile: ModelProfile) -> int:
    reserve = REASONING_RESERVE_TOKENS if profile.thinks else 0
    words = int(max(0, room - reserve) / OUTPUT_TOKENS_PER_WORD)
    return max(50, round(words, -1))


def _build_prompt(request: GenerationRequest, evidence: tuple[Chunk, ...]) -> str:
    from backend.app.verification.constraints import describe_constraints

    output = request.output
    deliverable = {
        key: value for key, value in asdict(output).items()
        if key not in {"original_request", "constraints"} and value not in (None, (), [], {})
    }
    return render_prompt(
        GENERATION_PROMPT,
        prompt_name="generation",
        USER_REQUEST=output.original_request or output.format,
        OUTPUT_SPECIFICATION=json.dumps(deliverable, ensure_ascii=False, sort_keys=True),
        MEASURED_REQUIREMENTS="\n".join(
            f"- {line}" for line in describe_constraints(output.constraints)
        ) or "None.",
        REVISION_FEEDBACK="\n".join(
            f"- {line}" for line in request.revision_feedback
        ) or "None. This is the first draft.",
        PREVIOUS_DRAFT=request.previous_draft or "None.",
        CURRENT_DRAFT=request.current_draft or "None.",
        CONVERSATION=request.conversation or "None. This is the first message.",
        SOURCE_EVIDENCE=render_source_evidence(evidence),
    )


def render_source_evidence(evidence: tuple[Chunk, ...]) -> str:
    """Present the source grouped by slide or section, one citable unit per line."""
    groups: dict[str, list[Chunk]] = {}
    for chunk in evidence:
        groups.setdefault(chunk.parent_id, []).append(chunk)
    rendered = []
    for chunks in groups.values():
        slide = next(
            (ref.slide_number for chunk in chunks for ref in chunk.evidence
             if ref.slide_number is not None),
            None,
        )
        title = next((chunk.text for chunk in chunks if chunk.chunk_type == "title"), None)
        opening = "<section>" if slide is None else (
            f'<slide number="{slide}"'
            + (f' title="{_attribute(title)}"' if title else "") + ">"
        )
        units = "\n".join(
            f'<evidence chunk_id="{chunk.chunk_id}" kind="{chunk.chunk_type}"'
            + (f' location="{location_label(chunk.evidence)}"'
               if slide is None and location_label(chunk.evidence) else "")
            + f"{_source_marker(chunk)}>\n{chunk.text}\n</evidence>"
            for chunk in chunks
        )
        rendered.append(f"{opening}\n{units}\n{'</section>' if slide is None else '</slide>'}")
    return "\n\n".join(rendered)


def _attribute(value: str) -> str:
    return " ".join(value.replace('"', "'").split())


def _source_marker(chunk: Chunk) -> str:
    """Label model-written image descriptions so they are never read as source text."""
    return ' source="image_description"' if chunk.evidence_kind == "derived" else ""


def _response_schema(evidence: tuple[Chunk, ...]) -> dict[str, Any]:
    support = {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string", "enum": [chunk.chunk_id for chunk in evidence]
            },
        },
        "required": ["chunk_id"],
        "additionalProperties": False,
    }
    unit = {
        "type": "object",
        "properties": {
            "unit_id": {"type": "string"},
            "text": {"type": "string"},
            "basis": {"type": "string", "enum": ["source", "background", "framing"]},
            "support": {"type": "array", "items": support},
            "contains_math": {"type": "boolean"},
            "role": {"type": "string"},
            "delivery": {"type": ["string", "null"]},
        },
        "required": [
            "unit_id", "role", "text", "basis", "support", "contains_math",
            "delivery"
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "blocks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "block_id": {"type": "string"},
                        "block_type": {"type": "string"},
                        "title": {"type": ["string", "null"]},
                        "units": {"type": "array", "minItems": 1, "items": unit},
                    },
                    "required": [
                        "block_id",
                        "block_type",
                        "title",
                        "units",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["blocks"],
        "additionalProperties": False,
    }


def _parse_blocks(
    payload: Mapping[str, Any], evidence: tuple[Chunk, ...]
) -> tuple[ArtifactBlock, ...]:
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("generation payload requires non-empty blocks")
    allowed = {
        chunk.chunk_id: {item.evidence_id for item in chunk.evidence}
        for chunk in evidence
    }
    blocks = []
    unit_ids: set[str] = set()
    block_ids: set[str] = set()
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise ValueError("each block must be an object")
        block_id = _required_string(raw, "block_id")
        if block_id in block_ids:
            raise ValueError("block IDs must be unique")
        block_ids.add(block_id)
        title = raw.get("title")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError("block title must be a non-empty string or null")
        blocks.append(
            ArtifactBlock(
                block_id=block_id,
                block_type=_required_string(raw, "block_type"),
                title=title.strip() if isinstance(title, str) else None,
                units=_parse_units(raw.get("units"), allowed, unit_ids),
            )
        )
    return tuple(blocks)


def _parse_units(raw_units, allowed, unit_ids) -> tuple[GroundedText, ...]:
    if not isinstance(raw_units, list):
        raise ValueError("content-unit collections must be arrays")
    units = []
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            raise ValueError("content units must be objects")
        unit_id = _required_string(raw, "unit_id")
        if unit_id in unit_ids:
            raise ValueError("content unit IDs must be globally unique")
        unit_ids.add(unit_id)
        basis = raw.get("basis")
        contains_math = raw.get("contains_math")
        if basis not in {"source", "background", "framing"} or not isinstance(contains_math, bool):
            raise ValueError("basis must be source, background or framing, and contains_math a boolean")
        raw_support = raw.get("support")
        if not isinstance(raw_support, list):
            raise ValueError("support must be an array")
        support = tuple(
            _parse_support(item, allowed) for item in raw_support
        ) if basis == "source" else ()
        delivery = raw.get("delivery")
        if delivery is not None and not isinstance(delivery, str):
            raise ValueError("delivery must be a string or null")
        units.append(
            GroundedText(
                unit_id=unit_id,
                role=_required_string(raw, "role"),
                text=_required_string(raw, "text"),
                factual=basis == "source",
                basis=basis,
                support=support,
                contains_math=contains_math,
                delivery=delivery,
            )
        )
    return tuple(units)


def _parse_support(raw, allowed) -> SupportReference:
    if not isinstance(raw, Mapping):
        raise ValueError("support references must be objects")
    chunk_id = _required_string(raw, "chunk_id")
    if chunk_id not in allowed:
        raise ValueError("generation cited evidence outside the supplied chunks")
    return SupportReference(chunk_id, tuple(sorted(allowed[chunk_id])))


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"generation field {key!r} must be a non-empty string")
    return value.strip()


def _validate_evidence_scope(
    request: GenerationRequest, evidence: tuple[Chunk, ...]
) -> None:
    if not evidence:
        raise ValueError("generation requires retrieved evidence")
    if any(
        chunk.document_id != request.document_id
        or chunk.document_version != request.document_version
        for chunk in evidence
    ):
        raise PermissionError("generation evidence is outside the selected document version")
