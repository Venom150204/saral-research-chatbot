"""Checking an answer the agent streamed in chat, right after it finishes."""

from __future__ import annotations

import re
from typing import Any

from backend.app.chunking import Chunk
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference,
)
from backend.app.verification import figure_findings

from .deck import source_ids

CITATION = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
MIN_WORDS_TO_CHECK = 25


def needs_check(text: str) -> bool:
    return bool(CITATION.search(text)) or bool(re.search(r"\d", text)) or len(text.split()) >= MIN_WORDS_TO_CHECK


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip().lstrip("-*• ").strip()
        if paragraph:
            sentences.extend(part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip())
    return sentences


def answer_units(text: str, chunks: tuple[Chunk, ...]) -> tuple[list[GroundedText], list[str]]:
    """Sentences as units: cited ones are claims about the work, the rest background.
    Also returns citations to source ids that do not exist."""
    ids = source_ids(chunks)
    units: list[GroundedText] = []
    unknown: list[str] = []
    for number, sentence in enumerate(split_sentences(text), start=1):
        cited: list[Chunk] = []
        for group in CITATION.findall(sentence):
            for sid in re.split(r"\s*[,;]\s*", group):
                if sid in ids and ids[sid] not in cited:
                    cited.append(ids[sid])
                elif sid not in ids:
                    unknown.append(sid)
        clean = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITATION.sub("", sentence).split()))
        if not clean or not re.search(r"\w", clean):
            continue
        support = tuple(
            SupportReference(chunk.chunk_id, tuple(ref.evidence_id for ref in chunk.evidence))
            for chunk in cited
        )
        units.append(GroundedText(
            f"answer-{number}", "answer", clean, bool(support), support,
            basis="source" if support else "background",
        ))
    return units, unknown


async def check_answer(
    *, text: str, chunks: tuple[Chunk, ...], verifier: Any,
    document_id: str, document_version: str, user_request: str, cited_only: bool = False,
) -> dict[str, Any] | None:
    """`cited_only`: after a tool produced or changed a draft (already checked by the
    pipeline), the reply is a status report; only its claims about the work, the
    sentences citing the deck, need checking."""
    units, unknown = answer_units(text, chunks)
    if cited_only:
        units = [unit for unit in units if unit.factual]
    if not units:
        return None
    artifact = GeneratedArtifact(
        "chat-answer", 1,
        GenerationRequest(document_id, document_version, OutputSpecification(
            "a chat answer", original_request=user_request,
        )),
        (ArtifactBlock("answer", "answer", tuple(units)),), "saral-agent", "chat-answer",
    )
    figures = [
        figure for finding in figure_findings(artifact, chunks)
        for figure in (*finding.unsupported, *finding.image_only)
    ]
    report = await verifier.verify(artifact, chunks)
    claims = list(report.verdicts)
    uncited = list(report.uncited_verdicts)
    issues = [f"{item.verdict.replace('_', ' ')}: {item.reason}" for item in report.rejected]
    issues += [f"cites {sid}, which is not in the deck" for sid in dict.fromkeys(unknown)]
    issues += [f"the figure {figure} is not in the deck" for figure in dict.fromkeys(figures)]
    return {
        "claims": len(claims),
        "claims_supported": sum(item.verdict == "supported" for item in claims),
        "explanations": len(uncited),
        "explanations_accepted": sum(item.verdict == "acceptable" for item in uncited),
        "issues": issues,
        "passed": not issues,
        "model_id": report.model_id,
    }
