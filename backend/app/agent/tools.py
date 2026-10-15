"""The agent's tools: thin wrappers around SARAL's existing, checked components.

The agent decides what to do; the tools guarantee how it is done. Every rule that
matters stays here, in code the model cannot skip: citations to real bullets, the
figure check, measured limits, the independent claim check, and versioned saves.

Each tool that saves claims a step in the operation ledger under its tool-call id,
so a retried or resumed call returns the stored result instead of saving twice.
Tools return a short summary for the model; the full result goes to the user's
screen as a stream event, not into the model's context.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from backend.app.chunking import location_label
from backend.app.editing import EditRejectedError, VerifiedEditService, resolve_artifact_target
from backend.app.generation import artifact_from_dict
from backend.app.retrieval import DeckRetriever
from backend.app.storage.conversations import TurnConflictError
from backend.app.verification import (
    measure_constraints, parse_constraints, validate_deterministic_support,
)
from backend.app.workflow.authoring import (
    StepResult, artifact_with_units, author, render_for_reader,
)

from .context import AgentContext, AgentServices
from .deck import source_ids

PREVIEW_CHARACTERS = 200
MAX_REQUIREMENTS = 8
CONVERSATION_CHARACTERS = 3_000


class Limit(BaseModel):
    """A measurable limit the user stated (or a format implies)."""

    kind: Literal["item_count", "max_characters", "max_words", "duration_seconds"]
    value: int = Field(description="The number; a spoken length in seconds")
    quote: str = Field(description="The user's exact words that set this limit")
    scope: Literal["whole", "each_item"] = "whole"
    basis: Literal["stated", "implied"] = Field(
        "stated", description="implied only for the known limit of a named format, e.g. a tweet"
    )


def _owner(ctx: AgentContext) -> dict[str, str]:
    return {"session_id": ctx.session_id, "tenant_id": ctx.tenant_id, "user_id": ctx.user_id}


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
    )


def _latest_user_message(runtime: ToolRuntime) -> str:
    for message in reversed(runtime.state.get("messages", [])):
        if getattr(message, "type", None) == "human":
            return _text(message)
    return ""


def _all_user_messages(runtime: ToolRuntime) -> str:
    return "\n".join(
        _text(message) for message in runtime.state.get("messages", [])
        if getattr(message, "type", None) == "human"
    )


def _recent_conversation(runtime: ToolRuntime) -> str | None:
    """User and assistant text from this conversation, newest kept, for the writer."""
    lines: list[str] = []
    total = 0
    for message in reversed(runtime.state.get("messages", [])):
        kind = getattr(message, "type", None)
        text = " ".join(_text(message).split())
        if kind not in {"human", "ai"} or not text:
            continue
        line = ("User: " if kind == "human" else "SARAL: ") + text
        if total + len(line) > CONVERSATION_CHARACTERS:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(reversed(lines)) or None


def summarize(result: dict[str, Any]) -> str:
    """What the model needs to know about a tool result, in a few hundred tokens."""
    value = result.get("value") or {}
    status = result.get("status")
    out: dict[str, Any] = {
        "status": {"succeeded": "saved", "blocked": "not saved", "failed_system": "failed"}.get(status, status)
    }
    artifact = value.get("artifact")
    if artifact:
        units = [unit for block in artifact.get("blocks", []) for unit in block.get("units", [])]
        text = " ".join(unit.get("text", "") for unit in units)
        out.update(
            version=artifact.get("version"), items=len(artifact.get("blocks", [])),
            words=len(text.split()), preview=text[:PREVIEW_CHARACTERS],
        )
    review = value.get("verification") or {}
    if review.get("verdicts") is not None:
        verdicts = review.get("verdicts", [])
        out["claims_supported"] = f"{sum(v['verdict'] == 'supported' for v in verdicts)} of {len(verdicts)}"
        requirements = review.get("requirement_verdicts") or []
        if requirements:
            out["requirements"] = [
                f"{'met' if item['met'] else 'NOT MET'}: {item['requirement']}"
                + ("" if item["met"] else f" ({item['reason']})")
                for item in requirements
            ]
        advice = [item for item in review.get("quality_issues") or [] if not item.startswith("Not met: ")]
        if review.get("fulfills_request") is False and advice:
            out["reviewer_advice"] = advice[:3]
    checked = (value.get("constraints") or {}).get("checked", [])
    if checked:
        out["limits"] = [
            f"{item['quote']}: {item['measured']} ({'met' if item['met'] else 'missed'})"
            for item in checked
        ]
    blocked = value.get("blocked") if isinstance(value.get("blocked"), dict) else (
        value if value.get("kind") == "blocked" else {}
    )
    if blocked:
        out["reason"] = blocked.get("message")
        issues = blocked.get("quality_issues") or []
        if issues:
            out["issues"] = list(issues)[:3]
    if status == "failed_system":
        out["reason"] = value.get("message")
    if value.get("deltas"):
        out["changed_units"] = len(value["deltas"])
        out["change_reason"] = value.get("reason")
    return json.dumps(out, ensure_ascii=False)


def build_tools(services: AgentServices) -> list:
    """The tool set, bound to the services it wraps."""

    def ledger_result(ctx: AgentContext, step_id: str) -> dict[str, Any] | None:
        existing = services.ledger.get_operation(**_owner(ctx), turn_id=ctx.turn_id, step_id=step_id)
        return existing.result if existing is not None and existing.status == "completed" else None

    def claim(ctx: AgentContext, step_id: str, capability: str) -> dict[str, Any] | None:
        """The stored result if this exact tool call already ran, else claim it."""
        done = ledger_result(ctx, step_id)
        if done is None:
            services.ledger.claim_operation(
                **_owner(ctx), turn_id=ctx.turn_id, step_id=step_id, capability=capability
            )
        return done

    def complete(ctx: AgentContext, step_id: str, capability: str, result: dict[str, Any]) -> None:
        if ledger_result(ctx, step_id) is None:
            services.ledger.complete_operation(
                **_owner(ctx), turn_id=ctx.turn_id, step_id=step_id,
                capability=capability, result=result,
            )

    def chunks_for(ctx: AgentContext):
        return services.document_repository.load_chunk_set(
            document_version=ctx.document_version, tenant_id=ctx.tenant_id, user_id=ctx.user_id,
        )

    def active_draft(ctx: AgentContext):
        active = services.artifact_repository.load_active(**_owner(ctx))
        return artifact_from_dict(active) if active else None

    def step_event(runtime: ToolRuntime, name: str):
        def emit(stage: str, detail: str) -> None:
            runtime.stream_writer({"type": "step", "tool": name, "stage": stage, "detail": detail})
        return emit

    @tool
    async def write_deliverable(
        deliverable: str,
        runtime: ToolRuntime[AgentContext],
        focus: str | None = None,
        limits: list[Limit] | None = None,
        requirements: list[str] | None = None,
        fix: str | None = None,
    ) -> str:
        """Create content the user will keep or use (a script, talk, slides, posts, an
        abstract, a summary) from the deck. It is written, measured, checked against the
        deck and saved as a new draft; the user sees it on screen.

        deliverable: what to make, for whom and in what style, e.g. "a 90-second
            script for policymakers on why this work matters".
        focus: the subject to concentrate on, in the user's words, if any.
        limits: limits the user stated, each with their exact words as `quote`
            ("90-second", "5 tweets", "under 280 characters"). Spoken length goes in
            seconds. Do not invent limits the user did not set.
        requirements: every other thing the user asked this deliverable to include or
            do, one short item each, close to their words ("a speaker note for each
            slide", "focus on the method", "for graduate students"). An independent
            reviewer judges each one; the result lists which were met.
        fix: only when redoing a deliverable from this turn whose requirements were
            not met: what the previous draft missed. The new draft revises it and
            replaces it on screen.
        """
        ctx, step = runtime.context, runtime.tool_call_id
        done = claim(ctx, step, "generate")
        if done is not None:
            return summarize(done)
        message = _latest_user_message(runtime)
        constraints, ignored = parse_constraints(
            [limit.model_dump() for limit in limits or []], _all_user_messages(runtime)
        )
        specification = deliverable if not fix else f"{deliverable}\nRevise the previous draft: {fix}"
        draft = active_draft(ctx) if fix else None
        result = asdict(await author(
            services=services, step_id=step, output_specification=specification,
            requirements=tuple(item.strip() for item in requirements or [] if item.strip())[:MAX_REQUIREMENTS],
            session_id=ctx.session_id, turn_id=ctx.turn_id, tenant_id=ctx.tenant_id,
            user_id=ctx.user_id, document_version=ctx.document_version,
            user_request=message, constraints=constraints, constraints_ignored=ignored,
            evidence_focus=focus or message,
            current_draft=render_for_reader(draft) if draft else ctx.current_draft,
            conversation=_recent_conversation(runtime),
            progress=step_event(runtime, "write_deliverable"),
        ))
        complete(ctx, step, "generate", result)
        runtime.stream_writer({"type": "artifact", "tool": "write_deliverable", "result": result,
                               "replaces_previous": bool(fix)})
        return summarize(result)

    @tool
    async def edit_draft(target: str, instruction: str, runtime: ToolRuntime[AgentContext]) -> str:
        """Change the wording of the saved current draft. Only the targeted part changes;
        the edit is checked against the deck and saved as a new version. It rewrites
        existing text only: it cannot add a missing part such as speaker notes or
        captions (use write_deliverable with `fix` for that).

        target: an exact block_id or unit_id from the current draft outline, or
            "artifact" for the whole draft.
        instruction: what to change, e.g. "make it simpler for teachers".
        """
        ctx, step = runtime.context, runtime.tool_call_id
        done = claim(ctx, step, "edit")
        if done is not None:
            return summarize(done)
        base = active_draft(ctx)
        if base is None:
            result = asdict(StepResult(step, "blocked", {
                "kind": "blocked", "category": "no_draft",
                "message": "There is no saved draft to edit yet.",
            }))
        else:
            emit = step_event(runtime, "edit_draft")
            emit("editing", f"Editing {target} and checking the changed part")
            message = _latest_user_message(runtime)
            full_instruction = instruction if instruction.strip() == message.strip() else (
                f"{instruction}\n(The user's own words: {message})"
            )
            chunk_set, _, _ = chunks_for(ctx)
            try:
                edit = await VerifiedEditService(
                    editor=services.editor, verifier=services.verifier,
                ).edit(artifact=base, target_id=target, instruction=full_instruction,
                       evidence=chunk_set.chunks)
            except EditRejectedError as error:
                result = asdict(StepResult(step, "blocked", {
                    "kind": "blocked", "category": "edit_rejected",
                    "message": error.user_message, "quality_issues": list(error.details),
                    "target_id": target,
                }))
            else:
                value = {
                    "kind": "edited_artifact", "artifact": asdict(edit.artifact),
                    "deltas": [asdict(delta) for delta in edit.deltas], "reason": edit.reason,
                    "constraints": {
                        "checked": [asdict(item) for item in measure_constraints(edit.artifact)],
                        "ignored": [],
                    },
                }
                result = asdict(StepResult(step, "succeeded", value))
                services.artifact_repository.save_revision(
                    artifact=edit.artifact, base_version=base.version, **_owner(ctx),
                    turn_id=ctx.turn_id, edit_instruction=instruction,
                    change_reason=edit.reason, operation_step_id=step, operation_result=result,
                )
                emit("saved", f"Saved version {edit.artifact.version}")
        complete(ctx, step, "edit", result)
        runtime.stream_writer({"type": "artifact", "tool": "edit_draft", "result": result})
        return summarize(result)

    @tool
    async def history(action: Literal["undo", "redo"], runtime: ToolRuntime[AgentContext]) -> str:
        """Undo or redo: move the current draft to the previous or next saved version."""
        ctx, step = runtime.context, runtime.tool_call_id
        done = claim(ctx, step, action)
        if done is not None:
            return summarize(done)
        try:
            restored = services.artifact_repository.move_history(
                direction=action, **_owner(ctx), turn_id=ctx.turn_id, operation_step_id=step,
            )
        except TurnConflictError as error:
            result = asdict(StepResult(step, "blocked", {
                "kind": "blocked", "category": "history",
                "message": f"Nothing to {action}." if "adjacent" in str(error) or "active" in str(error)
                else str(error),
            }))
            complete(ctx, step, action, result)
        else:
            result = asdict(StepResult(step, "succeeded", {
                "kind": action, "artifact": restored,
                "reason": f"Moved to version {restored['version']}.",
            }))
        runtime.stream_writer({"type": "artifact", "tool": "history", "result": result})
        return summarize(result)

    @tool
    async def check_claims(runtime: ToolRuntime[AgentContext], target: str = "artifact") -> str:
        """Check whether the current draft (or part of it) is supported by the deck.

        target: "artifact" for the whole draft, or a block_id / unit_id from the outline.
        """
        ctx = runtime.context
        base = active_draft(ctx)
        if base is None:
            return json.dumps({"status": "no draft", "reason": "There is no saved draft to check."})
        try:
            scoped = artifact_with_units(base, set(resolve_artifact_target(base, target)))
        except ValueError as error:
            return json.dumps({"status": "unknown target", "reason": str(error)})
        chunk_set, _, _ = chunks_for(ctx)
        step_event(runtime, "check_claims")("checking", "Checking the draft against the deck")
        problems = []
        try:
            validate_deterministic_support(scoped, chunk_set.chunks)
        except ValueError as error:
            problems.append(str(error))
        report = await services.verifier.verify(scoped, chunk_set.chunks)
        rejected = report.rejected
        runtime.stream_writer({"type": "check", "target": target, "report": {
            "verdicts": [asdict(item) for item in report.verdicts],
            "uncited_verdicts": [asdict(item) for item in report.uncited_verdicts],
            "problems": problems,
        }})
        return json.dumps({
            "status": "checked",
            "claims_supported": f"{sum(v.verdict == 'supported' for v in report.verdicts)} of {len(report.verdicts)}",
            "not_accepted": [f"{item.unit_id}: {item.verdict}: {item.reason}" for item in rejected][:5],
            "figure_problems": problems,
        }, ensure_ascii=False)

    @tool
    async def find_in_deck(query: str, runtime: ToolRuntime[AgentContext]) -> str:
        """Find the deck bullets that best match a question or phrase. Returns source ids
        with slide numbers. Use it to locate exact wording; no content is created."""
        ctx = runtime.context
        chunk_set, vectors, model_id = chunks_for(ctx)
        sids = {chunk.chunk_id: sid for sid, chunk in source_ids(chunk_set.chunks).items()}
        hits = await DeckRetriever(
            chunk_set, vectors, embedder=services.embedder, embedding_model_id=model_id,
        ).lookup(query, top_k=5)
        lines = []
        for item in hits:
            chunk = item.hit.chunk
            where = location_label(chunk.evidence) or "document"
            lines.append(f"[{sids[chunk.chunk_id]}] {where}: {' '.join(chunk.text.split())}")
        return "\n".join(lines) or "No matching bullets."

    @tool
    def ask_user(question: str, options: list[str]) -> str:
        """Ask the user one short question when the request is ambiguous in a way that
        changes the result. Give two to four short options to choose from."""
        answer = interrupt({"question": question, "options": [str(item) for item in options][:4]})
        return f"The user answered: {answer}"

    return [write_deliverable, edit_draft, history, check_claims, find_in_deck, ask_user]
