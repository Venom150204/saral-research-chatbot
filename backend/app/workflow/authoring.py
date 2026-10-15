"""One run of the writing pipeline (write, check, save or block), as a tool result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from .generation_graph import WorkflowStageError, build_generation_execution_graph

StepStatus = Literal["succeeded", "supported", "unsupported", "blocked", "failed_system", "skipped"]


@dataclass(frozen=True, slots=True)
class StepResult:
    """The outcome of one tool step, stored in the operation ledger and shown to the user."""

    step_id: str
    status: StepStatus
    value: Any = None


async def author(
    *, services, step_id: str, output_specification: str, requirements: tuple[str, ...],
    session_id: str, turn_id: str, tenant_id: str, user_id: str, document_version: str,
    user_request: str | None = None, constraints: tuple = (), constraints_ignored: tuple = (),
    evidence_focus: str | None = None, current_draft: str | None = None,
    conversation: str | None = None, progress=None,
) -> StepResult:
    """Write a deliverable from the deck, check it and save it, or block it with a reason."""
    graph = build_generation_execution_graph(
        document_repository=services.document_repository, embedder=services.embedder,
        generator=services.generator, verifier=services.verifier,
        artifact_repository=services.artifact_repository, editor=services.editor,
        progress=progress,
    )
    try:
        generated = await graph.ainvoke({
            "session_id": session_id, "turn_id": turn_id, "operation_id": step_id,
            "result_kind": "generated_artifact", "target_id": None,
            "tenant_id": tenant_id, "user_id": user_id, "document_version": document_version,
            "output_specification": output_specification, "requirements": requirements,
            "persist_artifact": True, "user_request": user_request,
            "constraints": constraints,
            "constraints_ignored": [dict(item) for item in constraints_ignored],
            "evidence_focus": evidence_focus, "current_draft": current_draft,
            "conversation": conversation,
        })
    except WorkflowStageError as error:
        if error.user_message:
            return StepResult(step_id, "blocked", {
                "kind": "blocked", "category": "request_too_long", "message": error.user_message,
            })
        return StepResult(step_id, "failed_system", {
            "kind": "system_failure", "stage": error.stage,
            "cause_type": error.cause_type, "retryable": error.retryable,
            "message": _PROVIDER_BUSY_MESSAGE if error.retryable else _safe_stage_message(error.stage),
        })
    report = generated.get("verification")
    value = {
        "kind": "generated_artifact",
        "target_id": None,
        "artifact": asdict(generated["artifact"]),
        "verification": None if report is None else {**asdict(report), "accepted": report.accepted},
        "selection_mode": generated["selection_mode"],
        "repair": generated.get("repair"),
        "regeneration": generated.get("regeneration"),
        "citation_completions": generated.get("citation_completions", []),
        "constraints": {
            "checked": generated.get("constraint_report", []),
            "ignored": generated.get("constraints_ignored", []),
        },
    }
    if generated.get("outcome_status") == "blocked":
        value["blocked"] = generated["blocked"]
        return StepResult(step_id, "blocked", value)
    return StepResult(step_id, "succeeded", value)


def render_for_reader(artifact) -> str:
    """The draft as the user sees it: numbered parts, titles and text."""
    return "\n\n".join(
        f"#{number}" + (f" {block.title}" if block.title else "") + "\n"
        + "\n".join(unit.text for unit in block.units)
        for number, block in enumerate(artifact.blocks, start=1)
    )


def artifact_with_units(artifact, allowed_ids: set[str]):
    """Only the chosen units of a draft, for checking one part of it."""
    blocks = tuple(
        replace(block, units=tuple(unit for unit in block.units if unit.unit_id in allowed_ids))
        for block in artifact.blocks
        if any(unit.unit_id in allowed_ids for unit in block.units)
    )
    output = replace(artifact.request.output, requirements=())
    return replace(artifact, blocks=blocks, request=replace(artifact.request, output=output))


_PROVIDER_BUSY_MESSAGE = (
    "The AI model was temporarily unavailable or over its request limit, even after "
    "several retries. Nothing was saved. This is safe to try again in a moment."
)


def _safe_stage_message(stage: str) -> str:
    return {
        "retrieve": "Evidence retrieval could not complete.",
        "generate": "The authoring model could not produce a valid draft.",
        "verify": "The verifier could not assess the draft.",
        "repair": "The bounded factual revision could not complete.",
        "regenerate": "The one automatic rewrite could not produce a valid draft.",
        "verify_after_repair": "Final verification could not complete.",
        "commit": "The accepted result could not be persisted.",
    }.get(stage, "The authoring operation could not complete.")
