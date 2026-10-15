"""Claim-level support verification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from backend.app.chunking import Chunk
from backend.app.generation import GeneratedArtifact


Verdict = Literal[
    "supported", "partially_supported", "unsupported", "contradicted", "insufficient_evidence"
]
UncitedVerdict = Literal["acceptable", "needs_source", "incorrect"]


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    unit_id: str
    verdict: Verdict | UncitedVerdict
    reason: str
    supporting_quotes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RequirementVerdict:
    """The reviewer's judgement of one thing the user asked for ("a speaker note for
    each slide"). Counts and lengths are measured by code instead."""

    requirement: str
    met: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SuggestedAction:
    label: str
    instruction: str


@dataclass(frozen=True, slots=True)
class UserGuidance:
    summary: str = ""
    observations: tuple[str, ...] = ()
    attempts: tuple[str, ...] = ()
    suggested_actions: tuple[SuggestedAction, ...] = ()
    can_retry_automatically: bool = False


@dataclass(frozen=True, slots=True)
class VerificationReport:
    verdicts: tuple[ClaimVerdict, ...]
    model_id: str
    prompt_version: str
    latency_ms: float
    fulfills_request: bool = True
    quality_reason: str = "Request-quality assessment was not supplied."
    quality_issues: tuple[str, ...] = ()
    guidance: UserGuidance = UserGuidance()
    uncited_verdicts: tuple[ClaimVerdict, ...] = ()
    requirement_verdicts: tuple[RequirementVerdict, ...] = ()

    @property
    def unmet_requirements(self) -> tuple[RequirementVerdict, ...]:
        return tuple(item for item in self.requirement_verdicts if not item.met)

    @property
    def rejected(self) -> tuple[ClaimVerdict, ...]:
        """Every unit that must be repaired: unsupported claims about the work, and
        uncited text that needs a source or is wrong."""
        return tuple(item for item in self.verdicts if item.verdict != "supported") + tuple(
            item for item in self.uncited_verdicts if item.verdict != "acceptable"
        )

    @property
    def accepted(self) -> bool:
        """Every claim about the work is supported and every uncited unit is genuine
        general knowledge or framing: the hard gate."""
        return not self.rejected


class SemanticSupportVerifier(Protocol):
    async def verify(
        self, artifact: GeneratedArtifact, evidence: tuple[Chunk, ...]
    ) -> VerificationReport: ...
