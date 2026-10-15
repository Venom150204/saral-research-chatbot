"""Deterministic evaluation metrics. Every function is plain arithmetic over labels,
so a reader can check each number by hand; model-judged metrics live in run_eval.py."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from typing import Iterable, Sequence

from backend.app.normalization import to_written_form


def normalize(text: str) -> str:
    """NFKC, case folding and collapsed whitespace: how labels are matched to text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def is_relevant(text: str, needles: Iterable[str]) -> bool:
    """A passage is answer-bearing when it contains any labelled answer phrase."""
    passage = normalize(text)
    return any(normalize(needle) in passage for needle in needles)


def ranking_scores(ranked: Sequence[str], needles: Sequence[str], total_relevant: int,
                   ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """Precision@k, Recall@k, F1@k, nDCG@k and reciprocal rank for one question."""
    flags = [is_relevant(text, needles) for text in ranked]
    scores: dict[str, float] = {}
    first = next((index for index, flag in enumerate(flags, start=1) if flag), None)
    scores["reciprocal_rank"] = 1 / first if first else 0.0
    for k in ks:
        found = sum(flags[:k])
        precision = found / k
        recall = found / total_relevant if total_relevant else 0.0
        scores[f"precision@{k}"] = precision
        scores[f"recall@{k}"] = recall
        scores[f"f1@{k}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(1 / math.log2(rank + 1) for rank, flag in enumerate(flags[:k], start=1) if flag)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, total_relevant) + 1))
        scores[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return scores


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {key: round(statistics.mean(row[key] for row in rows), 3) for key in rows[0]} if rows else {}


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Figures in a text, read in written form ("eighty point five" -> 80.5)."""
    return set(_NUMBER.findall(to_written_form(text)))


def numeric_overlap(sentence: str, evidence: str) -> bool | None:
    """Every figure in a sentence also appears in its cited evidence. None when the
    sentence has no figure (nothing to check)."""
    figures = numbers_in(sentence)
    return figures <= numbers_in(evidence) if figures else None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def percentile(values: Sequence[float], share: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


_ABSTAIN = re.compile(
    r"\b(does not|doesn't|do not|don't|not (?:mentioned|stated|specified|included|provided|reported|given|covered|say|mention|state|specify|include|provide|report|give|cover))\b",
    re.IGNORECASE,
)


def abstains(answer: str) -> bool:
    """A heuristic: the answer says the deck does not contain the information."""
    return bool(_ABSTAIN.search(answer))


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure with stemming (Google's rouge-score package)."""
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(reference, candidate)["rougeL"].fmeasure
