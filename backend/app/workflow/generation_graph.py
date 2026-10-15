"""Connected LangGraph branch for grounded generation and verified commit."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Any, Callable, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.app.embeddings import Embedder
from backend.app.editing import ArtifactEditor, validate_edit_result
from backend.app.generation import GenerationRequest, OutputSpecification
from backend.app.provider_retry import is_transient
from backend.app.retrieval import (
    DeckRetriever, document_fits_evidence_budget, select_generation_evidence,
)
from backend.app.verification import (
    SemanticSupportVerifier, UserGuidance, complete_numeric_citations,
    measure_constraints, validate_deterministic_support,
)


class Generator(Protocol):
    async def generate(self, request: GenerationRequest, evidence: tuple) -> Any: ...


class WorkflowStageError(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.retryable = is_transient(cause)
        self.user_message = getattr(cause, "user_message", None)
        super().__init__(f"{stage} stage failed: {cause}")


class GenerationExecutionState(TypedDict, total=False):
    session_id: str
    turn_id: str
    tenant_id: str
    user_id: str
    document_version: str
    output_specification: str
    user_request: str | None
    current_draft: str | None
    conversation: str | None
    constraints: tuple
    constraints_ignored: list[dict[str, Any]]
    operation_id: str
    result_kind: str
    target_id: str | None
    requirements: tuple[str, ...]
    evidence_focus: str | None
    evidence_override: tuple
    persist_artifact: bool
    evidence: tuple
    selection_mode: str
    artifact: Any
    draft_valid: bool
    draft_problems: list[str]
    constraint_report: list[dict[str, Any]]
    citation_completions: list[dict[str, str]]
    verification: Any
    repair: Any
    regeneration: Any
    committed: bool
    outcome_status: str
    blocked: dict[str, Any]


def _render_draft(artifact) -> str:
    """The draft as the writer returned it, so a rewrite can revise it."""
    return json.dumps([
        {
            "block_id": block.block_id, "block_type": block.block_type, "title": block.title,
            "units": [
                {
                    "unit_id": unit.unit_id, "role": unit.role, "text": unit.text,
                    "basis": unit.basis, "delivery": unit.delivery,
                    "support": [ref.chunk_id for ref in unit.support],
                }
                for unit in block.units
            ],
        }
        for block in artifact.blocks
    ], ensure_ascii=False)


def _enforced_unmet(state: GenerationExecutionState) -> list[dict[str, Any]]:
    """Exact limits the user stated that the current draft still misses. Targets
    such as spoken length are reported with the draft instead (`ENFORCED_KINDS`)."""
    return [
        item for item in state.get("constraint_report", [])
        if not item["met"] and item.get("enforced", item["basis"] == "stated")
    ]


def build_generation_execution_graph(
    *, document_repository, embedder: Embedder, generator: Generator,
    verifier: SemanticSupportVerifier, artifact_repository,
    editor: ArtifactEditor | None = None,
    progress: Callable[[str, str], None] | None = None,
):
    def notify(stage: str, detail: str) -> None:
        """Tell an observer (the agent's live step list) what the pipeline is doing.
        Reporting never changes or stops the work."""
        if progress is not None:
            try:
                progress(stage, detail)
            except Exception:
                pass

    def measured_summary(report_items) -> str:
        return "; ".join(
            f"{item['quote']}: {item['measured']}" + ("" if item["met"] else " (missed)")
            for item in report_items
        )

    async def select_evidence(state: GenerationExecutionState) -> dict:
        if state.get("evidence_override"):
            return {
                "evidence": state["evidence_override"],
                "selection_mode": "explicit_target_scope",
            }
        chunk_set, vectors, model_id = document_repository.load_chunk_set(
            document_version=state["document_version"],
            tenant_id=state["tenant_id"], user_id=state["user_id"],
        )
        ranked = ()
        if not document_fits_evidence_budget(chunk_set):
            try:
                ranked = await DeckRetriever(
                    chunk_set, vectors, embedder=embedder, embedding_model_id=model_id,
                ).lookup(
                    state.get("evidence_focus") or state["output_specification"],
                    top_k=min(30, len(chunk_set.chunks)),
                )
            except Exception as error:
                raise WorkflowStageError("retrieve", error) from error
        selection = select_generation_evidence(chunk_set, ranked)
        return {"evidence": selection.chunks, "selection_mode": selection.mode}

    def request_for(
        state: GenerationExecutionState, feedback=(), previous=None
    ) -> GenerationRequest:
        first = state["evidence"][0]
        return GenerationRequest(
            first.document_id, first.document_version,
            OutputSpecification(
                format=state["output_specification"],
                requirements=tuple(state.get("requirements", ())),
                original_request=state.get("user_request"),
                constraints=tuple(state.get("constraints", ())),
            ),
            revision_feedback=tuple(feedback),
            previous_draft=previous,
            current_draft=state.get("current_draft"),
            conversation=state.get("conversation"),
        )

    async def generate(state: GenerationExecutionState) -> dict:
        notify("writing", "Writing the draft")
        try:
            result = await generator.generate(request_for(state), state["evidence"])
        except Exception as error:
            raise WorkflowStageError("generate", error) from error
        return {"artifact": result.artifact}

    def ground(artifact, evidence, *, unit_ids=None):
        """Complete citations for figures the source holds, then apply the
        deterministic checks. Returns the grounded draft and what was added."""
        completed, added = complete_numeric_citations(
            artifact, evidence, unit_ids=unit_ids
        )
        validate_deterministic_support(completed, evidence)
        return completed, [asdict(item) for item in added]

    def measured(artifact) -> dict:
        """The limits measured on a valid draft, and what a rewrite must fix."""
        results = measure_constraints(artifact)
        return {
            "constraint_report": [asdict(item) for item in results],
            "draft_problems": [item.feedback for item in results if not item.met],
        }

    def check_draft(state: GenerationExecutionState) -> dict:
        """Free server checks, before any paid review: citations, figures, limits."""
        try:
            artifact, completions = ground(state["artifact"], state["evidence"])
        except ValueError as error:
            notify("measured", "The draft failed a citation or figure check")
            return {"draft_valid": False, "draft_problems": [str(error)]}
        result = measured(artifact)
        if result["constraint_report"]:
            notify("measured", "Measured: " + measured_summary(result["constraint_report"]))
        return {
            "artifact": artifact, "citation_completions": completions,
            "draft_valid": True, **result,
        }

    def route_after_check(state: GenerationExecutionState) -> str:
        return "regenerate" if state["draft_problems"] else "verify"

    async def regenerate(state: GenerationExecutionState) -> dict:
        """Rewrite the draft once, telling the writer exactly what was measured."""
        feedback = tuple(state["draft_problems"])
        notify("rewriting", "Revising the draft with the measured problems")
        try:
            result = await generator.generate(
                request_for(state, feedback, _render_draft(state["artifact"])),
                state["evidence"],
            )
        except Exception as error:
            raise WorkflowStageError("regenerate", error) from error
        try:
            rewritten, completions = ground(result.artifact, state["evidence"])
        except ValueError as error:
            if not state.get("draft_valid"):
                raise WorkflowStageError("regenerate", error) from error
            return {"regeneration": {
                "attempted": True, "produced_valid_draft": False,
                "feedback": list(feedback), "failure": str(error),
            }}
        return {
            "artifact": rewritten, "citation_completions": completions,
            "draft_valid": True, **measured(rewritten),
            "regeneration": {
                "attempted": True, "produced_valid_draft": True,
                "feedback": list(feedback),
            },
        }

    def route_after_regeneration(state: GenerationExecutionState) -> str:
        return "block" if _enforced_unmet(state) else "verify"

    async def verify(state: GenerationExecutionState) -> dict:
        units = [unit for block in state["artifact"].blocks for unit in block.units]
        claims = sum(unit.factual for unit in units)
        notify("checking", f"Checking {claims} claims against the deck and "
                           f"{len(units) - claims} other sentences")
        try:
            report = await verifier.verify(state["artifact"], state["evidence"])
        except Exception as error:
            stage = "verify_after_repair" if state.get("repair") else "verify"
            raise WorkflowStageError(stage, error) from error
        return {"verification": report}

    def route_after_verification(state: GenerationExecutionState) -> str:
        if not state["verification"].accepted:
            return "block" if state.get("repair") else "repair"
        return "block" if _enforced_unmet(state) else "commit"

    def block(state: GenerationExecutionState) -> dict:
        report = state.get("verification")
        if report is not None and not report.accepted:
            blocked = {
                "category": "support_rejected",
                "message": (
                    "Some statements are not fully supported by the source they "
                    "cite, and one targeted repair did not resolve it."
                ),
                "quality_issues": list(report.quality_issues),
                "guidance": asdict(report.guidance),
                "claim_issues": [
                    {"unit_id": item.unit_id, "verdict": item.verdict, "reason": item.reason}
                    for item in report.rejected
                ],
            }
        else:
            missed = [
                f"Required: {item['required']}. The draft has {item['measured']}."
                for item in _enforced_unmet(state)
            ]
            message = (
                "The draft does not meet a limit you set, even after one rewrite "
                "that was given the measured numbers."
            )
            blocked = {
                "category": "constraint_not_met",
                "message": message,
                "quality_issues": missed,
                "guidance": asdict(UserGuidance(summary=message, observations=tuple(missed))),
                "claim_issues": [],
            }
        notify("blocked", blocked["message"])
        return {"committed": False, "outcome_status": "blocked", "blocked": blocked}

    async def repair(state: GenerationExecutionState) -> dict:
        rejected = state["verification"].rejected
        notify("repairing", f"Repairing {len(rejected)} sentence(s) the check did not accept")
        failed = [f"{item.unit_id}:{item.verdict}" for item in rejected]
        if editor is None:
            raise WorkflowStageError(
                "repair", ValueError("artifact support verification failed: " + ", ".join(failed))
            )
        target_ids = tuple(item.unit_id for item in rejected)
        feedback = [
            {"unit_id": item.unit_id, "verdict": item.verdict, "reason": item.reason}
            for item in rejected
        ]
        instruction = (
            "Repair only the listed units. For a claim about this work that is not "
            "fully supported, remove, split, or weaken the unsupported portion so it "
            "follows directly from cited evidence. For uncited text marked needs_source, "
            "either cite the evidence it rests on (basis source) or remove what it says "
            "about this work. For uncited text marked incorrect, correct or remove it. "
            "Preserve the requested audience, format, other units, stable IDs, block "
            "structure, and overall length. Verification feedback: "
            + json.dumps(feedback, ensure_ascii=False)
        )
        try:
            result = await editor.edit(
                artifact=state["artifact"], target_unit_ids=target_ids,
                instruction=instruction, evidence=state["evidence"],
            )
            validate_edit_result(state["artifact"], result, set(target_ids))
            repaired, completions = ground(
                replace(result.artifact, version=1), state["evidence"],
                unit_ids=set(target_ids),
            )
        except Exception as error:
            raise WorkflowStageError("repair", error) from error
        return {
            "artifact": repaired,
            "citation_completions": [
                *state.get("citation_completions", []), *completions,
            ],
            **measured(repaired),
            "repair": {
                "attempted": True,
                "reason": result.reason,
                "deltas": [asdict(delta) for delta in result.deltas],
                "rejected_unit_ids": list(target_ids),
            },
        }

    def commit(state: GenerationExecutionState) -> dict:
        if not state.get("persist_artifact", True):
            return {"committed": False, "outcome_status": "completed"}
        stable_id = "art_" + sha256(
            f"{state['tenant_id']}:{state['session_id']}:{state['turn_id']}:{state.get('operation_id', 'authoring')}".encode()
        ).hexdigest()[:24]
        artifact = replace(state["artifact"], artifact_id=stable_id)
        verification = asdict(state["verification"])
        verification["accepted"] = state["verification"].accepted
        operation_result = {
            "step_id": state.get("operation_id", "authoring"),
            "status": "succeeded",
            "value": {
                "kind": state.get("result_kind", "generated_artifact"),
                "target_id": state.get("target_id"),
                "artifact": asdict(artifact),
                "verification": verification,
                "selection_mode": state["selection_mode"],
                "repair": state.get("repair"),
                "regeneration": state.get("regeneration"),
                "citation_completions": state.get("citation_completions", []),
                "constraints": {
                    "checked": state.get("constraint_report", []),
                    "ignored": state.get("constraints_ignored", []),
                },
            },
        }
        try:
            artifact_repository.save_initial(
                artifact=artifact, session_id=state["session_id"],
                turn_id=state["turn_id"], tenant_id=state["tenant_id"], user_id=state["user_id"],
                operation_step_id=state.get("operation_id"),
                operation_result=operation_result,
            )
        except Exception as error:
            raise WorkflowStageError("commit", error) from error
        notify("saved", f"Saved version {artifact.version}")
        return {"artifact": artifact, "committed": True, "outcome_status": "completed"}

    graph = StateGraph(GenerationExecutionState)
    graph.add_node("select_evidence", select_evidence)
    graph.add_node("generate", generate)
    graph.add_node("check_draft", check_draft)
    graph.add_node("regenerate", regenerate)
    graph.add_node("verify", verify)
    graph.add_node("repair", repair)
    graph.add_node("block", block)
    graph.add_node("commit", commit)
    graph.add_edge(START, "select_evidence")
    graph.add_edge("select_evidence", "generate")
    graph.add_edge("generate", "check_draft")
    graph.add_conditional_edges(
        "check_draft", route_after_check,
        {"verify": "verify", "regenerate": "regenerate"},
    )
    graph.add_conditional_edges(
        "regenerate", route_after_regeneration,
        {"verify": "verify", "block": "block"},
    )
    graph.add_conditional_edges(
        "verify", route_after_verification,
        {"commit": "commit", "repair": "repair", "block": "block"},
    )
    graph.add_edge("repair", "verify")
    graph.add_edge("block", END)
    graph.add_edge("commit", END)
    return graph.compile(checkpointer=False)
