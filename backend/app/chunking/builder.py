"""Slide-, heading- and equation-aware deterministic chunk construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import re

from backend.app.domain import ExtractedDocument, SourceBlock
from backend.app.normalization import to_written_form
from backend.app.visuals import DerivedVisualEvidence

from .models import Chunk, ChunkConfig, ChunkSet, EvidenceReference


def build_chunks(
    document: ExtractedDocument,
    *,
    visual_evidence: tuple[DerivedVisualEvidence, ...] = (),
    config: ChunkConfig | None = None,
) -> ChunkSet:
    active = config or ChunkConfig()
    _validate_visual_evidence(document, visual_evidence)
    if active.unit == "bullet":
        drafts = (
            _build_deck_unit_drafts(document, visual_evidence, active)
            if document.source_kind == "slide_deck"
            else _build_summary_unit_drafts(document, active)
        )
    else:
        drafts = (
            _build_slide_drafts(document, visual_evidence, active)
            if document.source_kind == "slide_deck"
            else _build_summary_drafts(document, active)
        )
    index_version = _index_version(document, active, visual_evidence)
    chunks = tuple(
        _make_chunk(document, active, index_version, ordinal, draft)
        for ordinal, draft in enumerate(drafts)
    )
    return ChunkSet(
        document.document_id, document.document_version, active, chunks, index_version
    )


def _index_version(document, config, visual_evidence) -> str:
    """Fingerprint what was extracted, how it was chunked, and which images were described."""
    blocks = sha256(
        "\0".join(block.block_id for block in document.blocks).encode("utf-8")
    ).hexdigest()
    payload = json.dumps(
        {
            "blocks": blocks,
            "config": asdict(config),
            "visual_evidence": sorted(item.evidence_id for item in visual_evidence),
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"idx_{sha256(payload).hexdigest()[:16]}"


def _searchable(context: str, text: str) -> str:
    """What is embedded and keyword-indexed for one unit."""
    context = context.strip()
    body = text if not context or context == text.strip() else f"{context}\n{text}"
    return to_written_form(body)


def _sentence_pieces(text: str, target_chars: int) -> list[str]:
    """Split a long block between sentences, packing up to `target_chars`."""
    if len(text) <= target_chars:
        return [text]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and size + addition > target_chars:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence) + (1 if size else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _unit_drafts(block: SourceBlock, chunk_type: str, context: str, config, *, split: bool = True) -> list[dict]:
    """One draft per block, or one per sentence group when the block is long.
    Every piece cites the same single source block. `split=False` keeps the block
    whole however long it is (a display equation must never be cut)."""
    pieces = _sentence_pieces(block.text, config.target_chars) if split else [block.text]
    return [
        {
            "parent_id": block.parent_id,
            "chunk_type": chunk_type,
            "evidence_kind": "native",
            "text": piece,
            "search_text": _searchable(context, piece),
            "evidence": (_source_ref(block),),
            "supports_quantitative_claims": True,
            "oversized": len(piece) > config.target_chars,
        }
        for piece in pieces
    ]


def _build_deck_unit_drafts(document, visual_evidence, config):
    """One citable unit per bullet, title, note, table, chart and described image."""
    boilerplate_ids = (
        _repeated_slide_boilerplate_ids(document)
        if config.filter_repeated_slide_boilerplate
        else set()
    )
    by_slide: dict[int, list[SourceBlock]] = defaultdict(list)
    for block in document.blocks:
        if block.location.slide_number is not None:
            by_slide[block.location.slide_number].append(block)
    derived_by_slide = defaultdict(list)
    for item in visual_evidence:
        derived_by_slide[item.slide_number].append(item)

    unit_types = {"title": "title", "slide_text": "bullet", "speaker_note": "speaker_note"}
    drafts = []
    for slide_number in sorted(set(by_slide) | set(derived_by_slide)):
        blocks = sorted(by_slide[slide_number], key=lambda item: item.ordinal)
        title = next((item.text for item in blocks if item.block_type == "title"), "")
        table_shapes: dict[int, list[SourceBlock]] = defaultdict(list)
        for block in blocks:
            if block.block_id in boilerplate_ids:
                continue
            if block.block_type in unit_types:
                drafts.extend(_unit_drafts(block, unit_types[block.block_type], title, config))
            elif block.block_type == "equation":
                drafts.extend(_unit_drafts(block, "equation", title, config, split=False))
            elif block.block_type == "chart_data":
                drafts.extend(_unit_drafts(block, "chart_data", title, config))
            elif block.block_type == "table_cell" and block.location.shape_id is not None:
                table_shapes[block.location.shape_id].append(block)
        for shape_id in sorted(table_shapes):
            for draft in _table_rows(document.filename, slide_number, title, table_shapes[shape_id], config):
                draft["search_text"] = _searchable(title, draft["text"])
                drafts.append(draft)
        for item in sorted(derived_by_slide[slide_number], key=lambda value: (value.shape_id, value.visual_id)):
            text = _derived_visual_text(item)
            drafts.append({
                "parent_id": blocks[0].parent_id if blocks else f"slide:{slide_number}",
                "chunk_type": "derived_visual",
                "evidence_kind": "derived",
                "text": text,
                "search_text": _searchable(title, text),
                "evidence": (EvidenceReference(item.evidence_id, "derived_visual", slide_number, item.shape_id),),
                "supports_quantitative_claims": not item.quantitative_claims_require_corroboration,
                "oversized": len(text) > config.target_chars,
            })
    return drafts


_TEXT_UNIT_TYPES = {"summary_paragraph": "paragraph", "paragraph": "paragraph",
                    "title": "title", "caption": "caption", "equation": "equation"}


def _build_summary_unit_drafts(document, config):
    """One citable unit per paragraph, with its section heading as context."""
    drafts = []
    heading: SourceBlock | None = None
    heading_used = True
    for block in sorted(document.blocks, key=lambda item: item.ordinal):
        if block.block_type == "heading":
            if heading is not None and not heading_used:
                drafts.extend(_unit_drafts(heading, "heading", "", config))
            heading, heading_used = block, False
            continue
        context = heading.text.lstrip("# ").strip() if heading is not None else ""
        drafts.extend(_unit_drafts(
            block, _TEXT_UNIT_TYPES.get(block.block_type, "paragraph"), context, config,
            split=block.block_type != "equation",
        ))
        heading_used = True
    if heading is not None and not heading_used:
        drafts.extend(_unit_drafts(heading, "heading", "", config))
    return drafts


def _derived_visual_text(item) -> str:
    parts = [item.literal_description]
    if item.visible_text:
        parts.append("Visible text: " + "; ".join(item.visible_text))
    if item.relationships:
        parts.append("Relationships: " + "; ".join(item.relationships))
    return "\n".join(parts)


def _build_slide_drafts(document, visual_evidence, config):
    boilerplate_ids = (
        _repeated_slide_boilerplate_ids(document)
        if config.filter_repeated_slide_boilerplate
        else set()
    )
    by_slide: dict[int, list[SourceBlock]] = defaultdict(list)
    for block in document.blocks:
        if block.location.slide_number is not None:
            by_slide[block.location.slide_number].append(block)
    derived_by_slide = defaultdict(list)
    for item in visual_evidence:
        derived_by_slide[item.slide_number].append(item)

    drafts = []
    for slide_number in sorted(set(by_slide) | set(derived_by_slide)):
        blocks = sorted(by_slide[slide_number], key=lambda item: item.ordinal)
        title = next((item.text for item in blocks if item.block_type == "title"), "")
        visible = [
            item
            for item in blocks
            if item.block_type in {"title", "slide_text"}
            and item.block_id not in boilerplate_ids
        ]
        drafts.extend(_group_blocks(document.filename, slide_number, title, "slide_text", visible, config))

        table_groups: dict[int, list[SourceBlock]] = defaultdict(list)
        for block in blocks:
            if block.block_type == "table_cell" and block.location.shape_id is not None:
                table_groups[block.location.shape_id].append(block)
        for shape_id in sorted(table_groups):
            drafts.extend(_table_rows(document.filename, slide_number, title, table_groups[shape_id], config))

        for block in blocks:
            if block.block_type == "chart_data":
                drafts.append(_block_draft(document.filename, slide_number, title, "chart_data", [block], config))

        notes = [item for item in blocks if item.block_type == "speaker_note"]
        drafts.extend(_group_blocks(document.filename, slide_number, title, "speaker_notes", notes, config))

        for item in sorted(derived_by_slide[slide_number], key=lambda value: (value.shape_id, value.visual_id)):
            text_parts = [item.literal_description]
            if item.visible_text:
                text_parts.append("Visible text: " + "; ".join(item.visible_text))
            if item.relationships:
                text_parts.append("Relationships: " + "; ".join(item.relationships))
            text = "\n".join(text_parts)
            prefix = _prefix(document.filename, slide_number, title, "derived_visual")
            drafts.append({
                "parent_id": f"slide:{slide_number}",
                "chunk_type": "derived_visual",
                "evidence_kind": "derived",
                "text": text,
                "search_text": f"{prefix}\n{text}",
                "evidence": (EvidenceReference(item.evidence_id, "derived_visual", slide_number, item.shape_id),),
                "supports_quantitative_claims": not item.quantitative_claims_require_corroboration,
                "oversized": len(text) > config.target_chars,
            })
    return drafts


def _repeated_slide_boilerplate_ids(document: ExtractedDocument) -> set[str]:
    """Find short template text repeated at identical geometry across most slides."""

    if document.unit_count < 3:
        return set()

    occurrences: dict[tuple[str, int, int, int, int], list[SourceBlock]] = defaultdict(list)
    for block in document.blocks:
        box = block.location.bounding_box
        if (
            block.block_type != "slide_text"
            or box is None
            or len(block.text.strip()) > 80
        ):
            continue
        normalized = re.sub(r"\s+", " ", block.text).strip().casefold()
        key = (normalized, box.left, box.top, box.width, box.height)
        occurrences[key].append(block)

    minimum_slides = max(3, (4 * document.unit_count + 4) // 5)
    boilerplate_ids: set[str] = set()
    for blocks in occurrences.values():
        slide_numbers = {item.location.slide_number for item in blocks}
        if len(slide_numbers) >= minimum_slides:
            boilerplate_ids.update(item.block_id for item in blocks)
    return boilerplate_ids


def _build_summary_drafts(document, config):
    drafts = []
    current_heading = ""
    group: list[SourceBlock] = []
    for block in sorted(document.blocks, key=lambda item: item.ordinal):
        if block.block_type == "heading":
            if group:
                drafts.extend(_summary_strategy_groups(document.filename, current_heading, group, config))
                group = []
            current_heading = block.text
        group.append(block)
    if group:
        drafts.extend(_summary_strategy_groups(document.filename, current_heading, group, config))
    return drafts


def _summary_strategy_groups(filename, heading, blocks, config):
    if config.summary_strategy == "recursive_structural":
        return _recursive_summary_groups(filename, heading, blocks, config)
    return _summary_groups(filename, heading, blocks, config)


def _summary_groups(filename, heading, blocks, config):
    results = []
    for group in _pack_at_boundaries(blocks, config.target_chars):
        text = "\n\n".join(item.text for item in group)
        refs = tuple(_source_ref(item) for item in group)
        prefix = f"Document: {filename}\nSection: {heading or '(unheaded)'}\nType: summary"
        results.append({
            "parent_id": group[0].parent_id,
            "chunk_type": "summary_section",
            "evidence_kind": "native",
            "text": text,
            "search_text": f"{prefix}\n{text}",
            "evidence": refs,
            "supports_quantitative_claims": True,
            "oversized": len(text) > config.target_chars,
        })
    return results


def _recursive_summary_groups(filename, heading, blocks, config):
    """Split long paragraphs at sentence boundaries, never inside a sentence."""

    units: list[tuple[str, SourceBlock]] = []
    for block in blocks:
        if len(block.text) <= config.target_chars or block.block_type == "heading":
            units.append((block.text, block))
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block.text)
            if sentence.strip()
        ]
        if len(sentences) == 1:
            units.append((block.text, block))
        else:
            units.extend((sentence, block) for sentence in sentences)

    packed: list[list[tuple[str, SourceBlock]]] = []
    current: list[tuple[str, SourceBlock]] = []
    size = 0
    for text, block in units:
        addition = len(text) + (2 if current else 0)
        if current and size + addition > config.target_chars:
            packed.append(current)
            current = []
            size = 0
        current.append((text, block))
        size += len(text) + (2 if size else 0)
    if current:
        packed.append(current)

    results = []
    prefix = f"Document: {filename}\nSection: {heading or '(unheaded)'}\nType: summary"
    for group in packed:
        text = "\n\n".join(item[0] for item in group)
        refs = _unique_refs(_source_ref(item[1]) for item in group)
        results.append({
            "parent_id": group[0][1].parent_id,
            "chunk_type": "summary_section",
            "evidence_kind": "native",
            "text": text,
            "search_text": f"{prefix}\n{text}",
            "evidence": refs,
            "supports_quantitative_claims": True,
            "oversized": len(text) > config.target_chars,
        })
    return results


def _group_blocks(filename, slide, title, chunk_type, blocks, config):
    return [
        _block_draft(filename, slide, title, chunk_type, group, config)
        for group in _pack_at_boundaries(blocks, config.target_chars)
    ]


def _block_draft(filename, slide, title, chunk_type, blocks, config):
    text = "\n".join(item.text for item in blocks)
    return {
        "parent_id": blocks[0].parent_id,
        "chunk_type": chunk_type,
        "evidence_kind": "native",
        "text": text,
        "search_text": f"{_prefix(filename, slide, title, chunk_type)}\n{text}",
        "evidence": tuple(_source_ref(item) for item in blocks),
        "supports_quantitative_claims": True,
        "oversized": len(text) > config.target_chars,
    }


def _table_rows(filename, slide, title, cells, config):
    rows: dict[int, list[SourceBlock]] = defaultdict(list)
    for cell in cells:
        if cell.location.table_row is not None:
            rows[cell.location.table_row].append(cell)
    if not rows:
        return []
    header_index = min(rows)
    header = sorted(rows[header_index], key=lambda item: item.location.table_column or 0)
    if config.table_layout == "grouped":
        return _grouped_table(filename, slide, title, rows, header_index, header, config)
    drafts = []
    for row_index in sorted(rows):
        row = sorted(rows[row_index], key=lambda item: item.location.table_column or 0)
        selected = header if row_index == header_index else [*header, *row]
        header_text = " | ".join(item.text for item in header)
        row_text = " | ".join(item.text for item in row)
        text = f"| {header_text} |\n| {' | '.join('---' for _ in header)} |"
        if row_index != header_index:
            text += f"\n| {row_text} |"
        prefix = _prefix(filename, slide, title, "table_row")
        drafts.append({
            "parent_id": row[0].parent_id,
            "chunk_type": "table_header" if row_index == header_index else "table_row",
            "evidence_kind": "native",
            "text": text,
            "search_text": f"{prefix}\n{text}",
            "evidence": tuple(_source_ref(item) for item in selected),
            "supports_quantitative_claims": True,
            "oversized": len(text) > config.target_chars,
        })
    return drafts


def _grouped_table(filename, slide, title, rows, header_index, header, config):
    """Keep table rows together under their header, splitting only between rows."""
    header_text = " | ".join(item.text for item in header)
    head = f"| {header_text} |\n| {' | '.join('---' for _ in header)} |"
    body = [
        sorted(rows[index], key=lambda item: item.location.table_column or 0)
        for index in sorted(rows)
        if index != header_index
    ]
    groups: list[list[list[SourceBlock]]] = []
    current: list[list[SourceBlock]] = []
    size = len(head)
    for row in body:
        line_length = len(" | ".join(item.text for item in row)) + 5
        if current and size + line_length > config.target_chars:
            groups.append(current)
            current, size = [], len(head)
        current.append(row)
        size += line_length
    if current or not groups:
        groups.append(current)
    prefix = _prefix(filename, slide, title, "table")
    drafts = []
    for group in groups:
        lines = [f"| {' | '.join(item.text for item in row)} |" for row in group]
        text = "\n".join([head, *lines])
        cells = [*header, *(cell for row in group for cell in row)]
        drafts.append({
            "parent_id": header[0].parent_id,
            "chunk_type": "table" if group else "table_header",
            "evidence_kind": "native",
            "text": text,
            "search_text": f"{prefix}\n{text}",
            "evidence": tuple(_source_ref(item) for item in cells),
            "supports_quantitative_claims": True,
            "oversized": len(text) > config.target_chars,
        })
    return drafts


def _pack_at_boundaries(blocks, target_chars):
    groups = []
    current = []
    size = 0
    for block in blocks:
        addition = len(block.text) + (1 if current else 0)
        if current and size + addition > target_chars:
            groups.append(current)
            current = []
            size = 0
        current.append(block)
        size += len(block.text) + (1 if size else 0)
    if current:
        groups.append(current)
    return groups


def _prefix(filename, slide, title, chunk_type):
    return f"Document: {filename}\nSlide: {slide}\nSlide title: {title or '(untitled)'}\nType: {chunk_type}"


def _source_ref(block):
    location = block.location
    return EvidenceReference(
        block.block_id,
        "source_block",
        location.slide_number,
        location.shape_id,
        location.line_start,
        location.line_end,
        location.page_number,
    )


def _unique_refs(refs):
    unique = []
    seen = set()
    for ref in refs:
        if ref.evidence_id not in seen:
            seen.add(ref.evidence_id)
            unique.append(ref)
    return tuple(unique)


def _make_chunk(document, config, index_version, ordinal, draft):
    evidence_ids = "\0".join(item.evidence_id for item in draft["evidence"])
    identity = "\0".join((document.document_version, index_version, config.version, config.summary_strategy, str(ordinal), draft["chunk_type"], evidence_ids, draft["text"])).encode("utf-8")
    return Chunk(
        chunk_id=f"chk_{sha256(identity).hexdigest()[:20]}",
        document_id=document.document_id,
        document_version=document.document_version,
        ordinal=ordinal,
        config_version=config.version,
        chunking_strategy=config.summary_strategy,
        **draft,
    )


def _validate_visual_evidence(document, evidence):
    visual_ids = {item.visual_id for item in document.visuals}
    for item in evidence:
        if item.document_id != document.document_id or item.document_version != document.document_version:
            raise ValueError("derived visual evidence belongs to another document version")
        if item.visual_id not in visual_ids:
            raise ValueError("derived visual evidence references an unknown visual")
