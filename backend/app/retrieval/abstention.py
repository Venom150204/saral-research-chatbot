"""Explicit similarity policy calibrated without held-out leakage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class LabeledScore:
    case_id: str
    score: float
    has_answer: bool


@dataclass(frozen=True, slots=True)
class SimilarityThresholdPolicy:
    threshold: float
    calibration_id: str
    objective: str = "zero development false accepts"

    def accepts(self, score: float | None) -> bool:
        return score is not None and score >= self.threshold


def calibrate_zero_false_accept_threshold(
    scores: Iterable[LabeledScore], *, calibration_id: str
) -> SimilarityThresholdPolicy:
    values = tuple(scores)
    negatives = [item.score for item in values if not item.has_answer]
    positives = [item.score for item in values if item.has_answer]
    if not negatives or not positives:
        raise ValueError("calibration requires positive and absent development cases")
    threshold = math.nextafter(max(negatives), math.inf)
    return SimilarityThresholdPolicy(threshold, calibration_id)


def evaluate_threshold(
    policy: SimilarityThresholdPolicy, scores: Iterable[LabeledScore]
) -> dict[str, float | int]:
    values = tuple(scores)
    positive = [item for item in values if item.has_answer]
    negative = [item for item in values if not item.has_answer]
    true_accept = sum(policy.accepts(item.score) for item in positive)
    false_accept = sum(policy.accepts(item.score) for item in negative)
    return {
        "positive_count": len(positive),
        "absent_count": len(negative),
        "positive_accept_rate": true_accept / len(positive) if positive else 0.0,
        "absent_false_accept_rate": false_accept / len(negative) if negative else 0.0,
        "accuracy": (true_accept + len(negative) - false_accept) / len(values)
        if values
        else 0.0,
    }
