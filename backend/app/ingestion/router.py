"""Filename-based upload routing at the ingestion boundary."""

from __future__ import annotations

from pathlib import Path

from backend.app.domain import ExtractedDocument
from backend.app.storage.assets import AssetStore

from .errors import UnsupportedFormatError
from .latex import extract_latex_bytes
from .pdf import extract_pdf_bytes
from .pptx import extract_pptx_bytes
from .summary import SUPPORTED_SUMMARY_SUFFIXES, extract_summary_bytes

SUPPORTED_UPLOAD_SUFFIXES = (".pptx", ".pdf", ".tex", *sorted(SUPPORTED_SUMMARY_SUFFIXES))


def extract_upload(
    payload: bytes, *, filename: str, asset_store: AssetStore | None = None
) -> ExtractedDocument:
    """Route an upload to its extractor."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pptx":
        return extract_pptx_bytes(payload, filename=filename, asset_store=asset_store)
    if suffix == ".pdf":
        return extract_pdf_bytes(payload, filename=filename)
    if suffix == ".tex":
        return extract_latex_bytes(payload, filename=filename)
    if suffix in SUPPORTED_SUMMARY_SUFFIXES:
        return extract_summary_bytes(payload, filename=filename)
    raise UnsupportedFormatError(
        f"unsupported upload format {suffix or '<none>'}; currently supported: "
        + ", ".join(SUPPORTED_UPLOAD_SUFFIXES)
    )
