"""Scoped lexical and dense retrieval baselines."""

from .dense import DenseIndex, cosine_similarity
from .abstention import (
    LabeledScore,
    SimilarityThresholdPolicy,
    calibrate_zero_false_accept_threshold,
    evaluate_threshold,
)
from .lexical import LexicalIndex, tokenize
from .models import SearchHit
from .hybrid import HybridHit, reciprocal_rank_fusion
from .retriever import Candidate, DeckRetriever
from .selection import (
    DEFAULT_EVIDENCE_CHARACTER_BUDGET,
    EvidenceSelection,
    document_fits_evidence_budget,
    select_generation_evidence,
)

__all__ = [
    "Candidate",
    "DEFAULT_EVIDENCE_CHARACTER_BUDGET",
    "DeckRetriever",
    "DenseIndex",
    "LabeledScore",
    "LexicalIndex",
    "SearchHit",
    "HybridHit",
    "reciprocal_rank_fusion",
    "EvidenceSelection",
    "document_fits_evidence_budget",
    "select_generation_evidence",
    "SimilarityThresholdPolicy",
    "calibrate_zero_false_accept_threshold",
    "cosine_similarity",
    "evaluate_threshold",
    "tokenize",
]
