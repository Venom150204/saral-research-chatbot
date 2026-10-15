"""Deck, paper and summary adapters (PPTX, PDF, LaTeX, Markdown, text) that preserve source provenance."""

from .errors import ExtractionError, ExtractionQualityError, UnsupportedFormatError
from .latex import extract_latex_bytes
from .pdf import extract_pdf_bytes
from .pptx import PptxExtractionLimits, extract_pptx_bytes
from .router import extract_upload
from .summary import extract_summary_bytes

__all__ = [
    "ExtractionError",
    "ExtractionQualityError",
    "PptxExtractionLimits",
    "UnsupportedFormatError",
    "extract_latex_bytes",
    "extract_pdf_bytes",
    "extract_pptx_bytes",
    "extract_summary_bytes",
    "extract_upload",
]
