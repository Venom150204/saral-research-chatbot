"""Gemini adapter for literal, provenance-safe visual descriptions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from google import genai
from google.genai import types

from backend.app.visuals.contracts import (
    VisualDescription,
    VisualDescriptionRequest,
    VisualDescriptionResult,
)
from backend.app.prompts import VISUAL_DESCRIPTION_PROMPT, render_prompt
from backend.app.provider_retry import call_with_retry

PROMPT_VERSION = "visual-literal-v1"
DEFAULT_MODEL = "gemini-3.8-flash"

VISUAL_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "literal_description": {"type": "string"},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "relationships": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": ["string", "null"]},
    },
    "required": [
        "literal_description",
        "visible_text",
        "entities",
        "relationships",
        "uncertainty",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GeminiTransport(Protocol):
    async def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        image_bytes: bytes,
        media_type: str,
        response_schema: Mapping[str, Any],
    ) -> ProviderResponse: ...


class GoogleGenAITransport:
    """Thin SDK boundary; the SDK owns transient wire encoding."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        image_bytes: bytes,
        media_type: str,
        response_schema: Mapping[str, Any],
    ) -> ProviderResponse:
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
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=dict(response_schema),
                    temperature=0,
                    max_output_tokens=1600,
                ),
            )

        payload = response.parsed
        if payload is None and response.text:
            payload = json.loads(response.text)
        if not isinstance(payload, Mapping):
            raise ValueError("Gemini returned no structured visual description")

        usage = response.usage_metadata
        return ProviderResponse(
            payload=payload,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )


class GeminiVisionDescriber:
    def __init__(
        self,
        transport: GeminiTransport,
        *,
        model_id: str = DEFAULT_MODEL,
    ) -> None:
        self._transport = transport
        self._model_id = model_id

    async def describe(
        self, request: VisualDescriptionRequest
    ) -> VisualDescriptionResult:
        prompt = _build_prompt(request)
        started = time.perf_counter()
        response = await self._generate_with_retry(prompt, request)
        latency_ms = (time.perf_counter() - started) * 1000
        payload = response.payload
        description = VisualDescription(
            visual_id=request.visual_id,
            model_id=self._model_id,
            prompt_version=PROMPT_VERSION,
            literal_description=_required_string(payload, "literal_description"),
            visible_text=_string_tuple(payload, "visible_text"),
            entities=_string_tuple(payload, "entities"),
            relationships=_string_tuple(payload, "relationships"),
            uncertainty=_optional_string(payload, "uncertainty"),
        )
        return VisualDescriptionResult(
            description=description,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )


    async def _generate_with_retry(
        self, prompt: str, request: VisualDescriptionRequest
    ) -> ProviderResponse:
        """Retry a transient provider failure; never retry an invalid response."""
        return await call_with_retry(lambda: self._transport.generate(
            model_id=self._model_id,
            prompt=prompt,
            image_bytes=request.asset.data,
            media_type=request.asset.media_type,
            response_schema=VISUAL_DESCRIPTION_SCHEMA,
        ))


def _build_prompt(request: VisualDescriptionRequest) -> str:
    return render_prompt(
        VISUAL_DESCRIPTION_PROMPT,
        prompt_name="visual_description",
        SLIDE_NUMBER=request.slide_number,
        SLIDE_CONTEXT=request.slide_context,
    )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Gemini response field {key!r} must be a non-empty string")
    return value.strip()


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Gemini response field {key!r} must be a string array")
    return tuple(item.strip() for item in value if item.strip())


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Gemini response field {key!r} must be a string or null")
    return value.strip() or None
