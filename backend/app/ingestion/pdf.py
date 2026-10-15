"""PDF extraction: a slide deck exported as PDF, or a paper or brief."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import statistics

from backend.app.domain import ExtractedDocument, ExtractionWarning, SourceBlock, SourceLocation

from .errors import ExtractionQualityError, UnsupportedFormatError

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 80
LANDSCAPE_RATIO = 1.15
MIN_CHARACTERS_PER_PAGE = 20
BULLET = re.compile(r"^\s*(?:[•▪◦●○■□‣∙·\-–—*]|\d{1,2}[.)])\s+")
PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.IGNORECASE)
HEADING = re.compile(r"^(?:(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+)?[A-Z][^.!?]{0,78}$")
NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+\S")
END_OF_BODY = re.compile(r"^(?:\d+\.?\s+)?(?:references|bibliography|works cited)$", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?$")


def extract_pdf_bytes(payload: bytes, *, filename: str) -> ExtractedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    safe_filename = Path(filename).name
    if not payload:
        raise ExtractionQualityError("PDF is empty")
    if len(payload) > MAX_PDF_BYTES:
        raise UnsupportedFormatError(f"PDF is {len(payload)} bytes; limit is {MAX_PDF_BYTES} bytes")
    try:
        reader = PdfReader(BytesIO(payload))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnsupportedFormatError("PDF is password-protected")
        pages = list(reader.pages)
    except PdfReadError as error:
        raise UnsupportedFormatError(f"PDF could not be read: {error}") from error
    if not pages:
        raise ExtractionQualityError("PDF has no pages")
    if len(pages) > MAX_PDF_PAGES:
        raise UnsupportedFormatError(f"PDF has {len(pages)} pages; limit is {MAX_PDF_PAGES}")

    texts = [_page_lines(page.extract_text() or "") for page in pages]
    if sum(len(" ".join(lines)) for lines in texts) < MIN_CHARACTERS_PER_PAGE * len(pages):
        raise ExtractionQualityError(
            "PDF has almost no text layer (it looks scanned); OCR is not supported"
        )
    shapes = [float(page.mediabox.width) / max(float(page.mediabox.height), 1.0) for page in pages]
    is_deck = sum(ratio >= LANDSCAPE_RATIO for ratio in shapes) > len(pages) / 2
    repeated = _repeated_lines(texts)
    warnings = []
    if repeated:
        warnings.append(ExtractionWarning(
            "repeated_page_text_skipped",
            f"Left out text repeated on most pages (headers, footers): {'; '.join(sorted(repeated))[:300]}",
        ))
    content_hash = sha256(payload).hexdigest()
    kept = [[line for line in lines if _key(line) not in repeated and not PAGE_NUMBER.match(line)]
            for lines in texts]
    if is_deck:
        blocks = _deck_blocks(kept, content_hash)
        kind, units = "slide_deck", len(pages)
    else:
        blocks, stopped = _document_blocks(kept, content_hash)
        kind, units = "document", len(pages)
        if stopped:
            warnings.append(ExtractionWarning(
                "references_skipped", f"The reference list from page {stopped} on is not indexed.",
            ))
    if not blocks:
        raise ExtractionQualityError("no readable text was found in the PDF")
    return ExtractedDocument(
        document_id=f"doc_{content_hash[:16]}",
        document_version=f"sha256:{content_hash}",
        content_hash=content_hash,
        filename=safe_filename,
        media_type="application/pdf",
        source_kind=kind,
        parser_name="pypdf",
        parser_version=_pypdf_version(),
        unit_count=units,
        blocks=tuple(blocks),
        warnings=tuple(warnings),
    )


def _pypdf_version() -> str:
    import pypdf

    return pypdf.__version__


def _page_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _key(line: str) -> str:
    """Repeated-line identity: digits masked so "Page 3" and "Page 4" match."""
    return re.sub(r"\d+", "#", line.casefold())


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    if len(pages) < 3:
        return set()
    counts = Counter(key for lines in pages for key in {_key(line) for line in lines})
    threshold = max(3, round(0.6 * len(pages)))
    return {key for key, count in counts.items() if count >= threshold and len(key) <= 120}


def _block(content_hash: str, parent: str, ordinal: int, text: str, block_type: str,
           location: SourceLocation, level: int | None = None) -> SourceBlock:
    identity = f"{content_hash}\0{ordinal}\0{block_type}\0{text}".encode("utf-8")
    return SourceBlock(
        block_id=f"blk_{sha256(identity).hexdigest()[:20]}", parent_id=parent, ordinal=ordinal,
        text=text, location=location, block_type=block_type, hierarchy_level=level,
    )


def _deck_blocks(pages: list[list[str]], content_hash: str) -> list[SourceBlock]:
    """One slide per page: the first line is the title, the rest are bullets.
    A line continues the previous bullet unless it starts with a bullet mark or the
    previous one ended a sentence."""
    blocks: list[SourceBlock] = []
    for number, lines in enumerate(pages, start=1):
        if not lines:
            continue
        parent = f"pdfslide_{content_hash[:12]}_{number}"
        location = SourceLocation(slide_number=number, page_number=number)
        blocks.append(_block(content_hash, parent, len(blocks), lines[0], "title", location))
        bullets: list[str] = []
        for line in lines[1:]:
            starts_bullet = bool(BULLET.match(line))
            text = BULLET.sub("", line).strip()
            if not text:
                continue
            if bullets and not starts_bullet and bullets[-1].endswith("-") and text[:1].islower():
                bullets[-1] = bullets[-1][:-1] + text
            elif bullets and not starts_bullet and not SENTENCE_END.search(bullets[-1]) and text[:1].islower():
                bullets[-1] = f"{bullets[-1]} {text}"
            else:
                bullets.append(text)
        for text in bullets:
            blocks.append(_block(content_hash, parent, len(blocks), text, "slide_text", location, 0))
    return blocks


def _document_blocks(pages: list[list[str]], content_hash: str) -> tuple[list[SourceBlock], int | None]:
    """Paragraphs under headings, each cited by the page it starts on."""
    lengths = [len(line) for lines in pages for line in lines]
    typical = statistics.median(lengths) if lengths else 80
    blocks: list[SourceBlock] = []
    section = 0
    parent = f"pdfsec_{content_hash[:12]}_{section}"
    paragraph: list[str] = []
    start_page = 1

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            text = paragraph[0]
            for line in paragraph[1:]:
                text = text[:-1] + line if text.endswith("-") and line[:1].islower() else f"{text} {line}"
            blocks.append(_block(content_hash, parent, len(blocks), text, "paragraph",
                                 SourceLocation(page_number=start_page)))
        paragraph = []

    for number, lines in enumerate(pages, start=1):
        for index, line in enumerate(lines):
            if END_OF_BODY.match(line):
                flush()
                return blocks, number
            if number == 1 and not blocks and not paragraph and index == 0 and len(line) <= 150:
                blocks.append(_block(content_hash, parent, 0, line, "title", SourceLocation(page_number=1)))
                continue
            if _is_heading(line, typical):
                flush()
                section += 1
                parent = f"pdfsec_{content_hash[:12]}_{section}"
                blocks.append(_block(content_hash, parent, len(blocks), line, "heading",
                                     SourceLocation(page_number=number)))
                continue
            if not paragraph:
                start_page = number
            paragraph.append(line)
            if SENTENCE_END.search(line) and len(line) < 0.75 * typical:
                flush()
    flush()
    return blocks, None


def _is_heading(line: str, typical: float) -> bool:
    if len(line) > 80 or SENTENCE_END.search(line) or not HEADING.match(line):
        return False
    if line.casefold() in {"abstract", "introduction", "conclusion", "conclusions", "summary",
                           "background", "methods", "results", "discussion", "key findings",
                           "recommendations", "executive summary"}:
        return True
    return bool(NUMBERED_HEADING.match(line)) and len(line.split()) <= 10 and len(line) < 0.8 * typical
