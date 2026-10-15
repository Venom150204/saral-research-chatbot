"""Deterministic handling of the figures a writer puts in a factual sentence."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable

from backend.app.chunking import Chunk
from backend.app.generation import GeneratedArtifact, GroundedText, SupportReference
from backend.app.normalization import to_written_form

_FIGURE = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")
_SOURCE_LOCATOR = re.compile(
    r"\b(?:slide|page|figure|fig\.?|table|equation|eq\.?)\s*#?\s*\d+\b", re.IGNORECASE
)
_ITEM_LABEL = re.compile(
    r"^\W*(?:[A-Za-z][\w-]*\s+){0,3}#?(\d+)(?:\s*(?:/|of)\s*\d+)?"
    r"\s*[:.)\]–—-](?=\s|$)"
)


_WORD_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")


def _extract(written: str) -> tuple[tuple[str, Decimal], ...]:
    """Figures as (shown the way they were written, value). A percent sign is kept
    for display and ignored for the value: the unit is the verifier's to judge."""
    return tuple(
        (
            match.group(0) + ("%" if written[match.end():match.end() + 1] == "%" else ""),
            Decimal(match.group(0).replace(",", "")).normalize(),
        )
        for match in _FIGURE.finditer(written)
    )


def claim_figures(text: str, *, item_count: int) -> tuple[tuple[str, Decimal], ...]:
    """The figures a writer asserted, as (how it was written, its value)."""
    written = _SOURCE_LOCATOR.sub("", to_written_form(text, convert_number_words=False))
    label = _ITEM_LABEL.match(written)
    if label is not None and 0 < int(label.group(1)) <= item_count:
        written = written[label.end():]
    return _extract(written)


def source_values(texts: Iterable[str]) -> set[Decimal]:
    """Every value the source could be stating. Read generously: number words,
    spoken decimals and lone small numbers all count, and so does a number word
    joined to another word ("seven-slide", "thirty-four-layer")."""
    return {
        value
        for text in texts
        for reading in (text, _WORD_HYPHEN.sub(" ", text))
        for _, value in _extract(to_written_form(reading, convert_isolated_small_numbers=True))
    }


@dataclass(frozen=True, slots=True)
class CitationCompletion:
    """A citation the server added because the cited figure sits in that unit."""

    unit_id: str
    chunk_id: str
    figure: str


@dataclass(frozen=True, slots=True)
class FigureFindings:
    """What the numeric check found for one factual unit."""

    unit_id: str
    unsupported: tuple[str, ...] = ()
    image_only: tuple[str, ...] = ()
    cited_elsewhere: tuple[str, ...] = ()


def _native(chunks: Iterable[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.supports_quantitative_claims]


def _request_text(artifact: GeneratedArtifact) -> str:
    """What the user asked for. Server-written requirements and rewrite feedback
    are left out: a figure the server rejected must not become "requested"."""
    output = artifact.request.output
    return "\n".join(
        part for part in (
            output.original_request, output.format, output.audience, output.purpose,
            output.style_guidance,
        ) if part
    )


def complete_numeric_citations(
    artifact: GeneratedArtifact,
    evidence: tuple[Chunk, ...],
    *,
    unit_ids: set[str] | None = None,
) -> tuple[GeneratedArtifact, tuple[CitationCompletion, ...]]:
    """Add the citation a factual unit left out for a figure the source does contain."""
    by_id = {chunk.chunk_id: chunk for chunk in evidence}
    holders: dict[Decimal, list[Chunk]] = {}
    for chunk in _native(evidence):
        for value in source_values([chunk.text]):
            holders.setdefault(value, []).append(chunk)
    item_count = sum(len(block.units) for block in artifact.blocks)
    completions: list[CitationCompletion] = []

    def complete(unit: GroundedText) -> GroundedText:
        if not unit.factual or (unit_ids is not None and unit.unit_id not in unit_ids):
            return unit
        cited = [by_id[ref.chunk_id] for ref in unit.support if ref.chunk_id in by_id]
        cited_ids = {chunk.chunk_id for chunk in cited}
        available = source_values(chunk.text for chunk in _native(cited))
        cited_parents = {chunk.parent_id for chunk in cited}
        added: list[Chunk] = []
        for written, value in claim_figures(unit.text, item_count=item_count):
            if value in available or value not in holders:
                continue
            candidates = [c for c in holders[value] if c.chunk_id not in cited_ids]
            nearby = [c for c in candidates if c.parent_id in cited_parents]
            chosen = nearby or (candidates if len(holders[value]) == 1 else [])
            for chunk in chosen:
                added.append(chunk)
                cited_ids.add(chunk.chunk_id)
                completions.append(CitationCompletion(unit.unit_id, chunk.chunk_id, written))
            available |= source_values(chunk.text for chunk in chosen)
        if not added:
            return unit
        return replace(unit, support=unit.support + tuple(
            SupportReference(
                chunk.chunk_id, tuple(sorted(ref.evidence_id for ref in chunk.evidence))
            )
            for chunk in added
        ))

    blocks = tuple(
        replace(block, units=tuple(complete(unit) for unit in block.units))
        for block in artifact.blocks
    )
    if not completions:
        return artifact, ()
    return replace(artifact, blocks=blocks), tuple(completions)


def figure_findings(
    artifact: GeneratedArtifact, evidence: tuple[Chunk, ...]
) -> tuple[FigureFindings, ...]:
    """Classify every figure in every factual unit. Units with nothing to report
    are omitted. Citations are assumed to exist; scope is checked elsewhere."""
    by_id = {chunk.chunk_id: chunk for chunk in evidence}
    in_source = source_values(chunk.text for chunk in _native(evidence))
    in_request = source_values([_request_text(artifact)])
    item_count = sum(len(block.units) for block in artifact.blocks)
    findings = []
    for block in artifact.blocks:
        for unit in block.units:
            if not unit.factual:
                continue
            cited = [by_id[ref.chunk_id] for ref in unit.support if ref.chunk_id in by_id]
            cited_values = source_values(chunk.text for chunk in _native(cited))
            image_values = source_values(
                chunk.text for chunk in cited if not chunk.supports_quantitative_claims
            )
            unsupported, image_only, elsewhere = [], [], []
            for written, value in claim_figures(unit.text, item_count=item_count):
                if value in cited_values:
                    continue
                if value in in_source:
                    elsewhere.append(written)
                elif value in image_values:
                    image_only.append(written)
                elif value in in_request:
                    continue
                else:
                    unsupported.append(written)
            if unsupported or image_only or elsewhere:
                findings.append(FigureFindings(
                    unit.unit_id, tuple(unsupported), tuple(image_only), tuple(elsewhere)
                ))
    return tuple(findings)
