"""Plain-text and Markdown summary extraction with line provenance."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from backend.app.domain import ExtractedDocument, SourceBlock, SourceLocation

from .errors import ExtractionQualityError, UnsupportedFormatError

SUPPORTED_SUMMARY_SUFFIXES = {".txt", ".md", ".markdown"}
MAX_SUMMARY_BYTES = 2 * 1024 * 1024


def extract_summary_bytes(payload: bytes, *, filename: str) -> ExtractedDocument:
    safe_filename = Path(filename).name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_SUMMARY_SUFFIXES:
        raise UnsupportedFormatError(
            f"summary format {suffix or '<none>'} is unsupported"
        )
    if not payload:
        raise ExtractionQualityError("summary is empty")
    if len(payload) > MAX_SUMMARY_BYTES:
        raise UnsupportedFormatError(
            f"summary is {len(payload)} bytes; limit is {MAX_SUMMARY_BYTES} bytes"
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedFormatError("summary must be UTF-8 encoded") from exc
    if not text.strip():
        raise ExtractionQualityError("summary contains only whitespace")

    content_hash = sha256(payload).hexdigest()
    parent_id = f"sum_{content_hash[:16]}"
    blocks: list[SourceBlock] = []
    lines = text.splitlines(keepends=True)
    paragraph_lines: list[str] = []
    paragraph_start = 1

    def flush(line_end: int) -> None:
        nonlocal paragraph_lines, paragraph_start
        if not paragraph_lines:
            return
        paragraph = "".join(paragraph_lines).rstrip("\r\n")
        if paragraph.strip():
            block_type = (
                "heading"
                if suffix in {".md", ".markdown"}
                and paragraph.lstrip().startswith("#")
                else "summary_paragraph"
            )
            ordinal = len(blocks)
            identity = (
                f"{content_hash}\0{paragraph_start}\0{line_end}\0{paragraph}"
            ).encode("utf-8")
            blocks.append(
                SourceBlock(
                    block_id=f"blk_{sha256(identity).hexdigest()[:20]}",
                    parent_id=parent_id,
                    ordinal=ordinal,
                    text=paragraph,
                    block_type=block_type,
                    location=SourceLocation(
                        line_start=paragraph_start,
                        line_end=line_end,
                    ),
                )
            )
        paragraph_lines = []

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not paragraph_lines:
                paragraph_start = line_number
            paragraph_lines.append(line)
        else:
            flush(line_number - 1)
    flush(len(lines))

    return ExtractedDocument(
        document_id=f"doc_{content_hash[:16]}",
        document_version=f"sha256:{content_hash}",
        content_hash=content_hash,
        filename=safe_filename,
        media_type="text/markdown" if suffix != ".txt" else "text/plain",
        source_kind="summary",
        parser_name="saral-summary",
        parser_version="1",
        unit_count=len(blocks),
        blocks=tuple(blocks),
    )
