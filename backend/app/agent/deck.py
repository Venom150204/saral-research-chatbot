"""The source as the agent sees it: every citable unit with a short source id."""

from __future__ import annotations

from typing import Iterable

from backend.app.chunking import Chunk, location_label


def source_ids(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    """S1..Sn in deck order."""
    return {f"S{number}": chunk for number, chunk in enumerate(chunks, start=1)}


def _group(chunk: Chunk) -> str:
    """Slides and PDF pages group units; a LaTeX or text source is one document,
    each unit labelled with its lines."""
    label = location_label(chunk.evidence) or ""
    return label.capitalize() if label.startswith(("slide", "page")) else "Document"


def render_deck(chunks: Iterable[Chunk]) -> str:
    """Grouped by slide or page, one line per citable unit."""
    lines: list[str] = []
    current: object = object()
    for sid, chunk in source_ids(chunks).items():
        group = _group(chunk)
        if group != current:
            lines.append(group)
            current = group
        marker = " (image description: not the authors' words)" if chunk.evidence_kind == "derived" else ""
        if group == "Document" and location_label(chunk.evidence):
            marker += f" ({location_label(chunk.evidence)})"
        lines.append(f"  [{sid}] {' '.join(chunk.text.split())}{marker}")
    return "\n".join(lines)
