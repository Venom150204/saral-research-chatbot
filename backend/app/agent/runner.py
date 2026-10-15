"""Run one agent turn and stream it as simple events for the browser."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from hashlib import sha256
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from backend.app.editing import artifact_outline
from backend.app.generation import artifact_from_dict
from backend.app.workflow.authoring import render_for_reader

from .answer_check import CITATION, check_answer, needs_check
from .context import AgentContext, AgentServices
from .deck import render_deck, source_ids
from .graph import PROMPT_VERSION

logger = logging.getLogger(__name__)

AGENT_PLAN = {
    "request_type": "agent",
    "operations": [{"operation_id": "conversation", "description": "agent turn"}],
    "steps": [
        {"step_id": f"agent-{capability}", "capability": capability, "target_id": None,
         "parameters": {}, "operation_ids": ["conversation"], "depends_on": [], "condition": "always"}
        for capability in ("generate", "edit", "undo", "redo")
    ],
    "clarification": None,
}
RECURSION_LIMIT = 100
LIMIT_REACHED = (
    "I stopped here to keep this turn within its limits. Tell me what to do next, or "
    "split the request into smaller steps."
)

_TOOL_LABELS = {
    "write_deliverable": "Starting the draft",
    "edit_draft": "Editing the draft",
    "history": "Moving between versions",
    "check_claims": "Checking the draft",
    "find_in_deck": "Looking in the deck",
    "ask_user": "Asking you a question",
}


class FinalDrafts:
    """The drafts a turn shows: the latest version of each, in order of first save."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, event: dict[str, Any]) -> None:
        if event.get("replaces_previous"):
            for index in range(len(self.items) - 1, -1, -1):
                if self.items[index]["tool"] in {"write_deliverable", "edit_draft"}:
                    del self.items[index]
                    break
        artifact_id = _artifact_id(event)
        for index, item in enumerate(self.items):
            if artifact_id is not None and _artifact_id(item) == artifact_id:
                self.items[index] = event
                return
        self.items.append(event)

    def results(self) -> list[dict[str, Any]]:
        return [item["result"] for item in self.items]


def _artifact_id(event: dict[str, Any]) -> str | None:
    artifact = ((event.get("result") or {}).get("value") or {}).get("artifact") or {}
    return artifact.get("artifact_id")


def thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"agent:{session_id}"}, "recursion_limit": RECURSION_LIMIT}


def _chunk_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
    )


async def stream_turn(
    *,
    agent: Any,
    services: AgentServices,
    agent_model_id: str,
    session_id: str,
    turn_id: str,
    tenant_id: str,
    user_id: str,
    document_id: str,
    document_version: str,
    message: str | None = None,
    resume: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    owner = {"session_id": session_id, "tenant_id": tenant_id, "user_id": user_id}
    started = time.perf_counter()
    ledger = services.ledger
    if resume is None:
        if not (message or "").strip():
            yield {"type": "error", "message": "Type a message first."}
            return
        ledger.begin_turn(
            **owner, turn_id=turn_id, message=message,
            request_sha256=sha256(message.encode("utf-8")).hexdigest(),
        )
        ledger.save_turn_plan(
            **owner, turn_id=turn_id, plan=AGENT_PLAN,
            model_id=agent_model_id, prompt_version=PROMPT_VERSION,
        )

    chunk_set, _, _ = services.document_repository.load_chunk_set(
        document_version=document_version, tenant_id=tenant_id, user_id=user_id,
    )
    active = services.artifact_repository.load_active(**owner)
    draft = artifact_from_dict(active) if active else None
    context = AgentContext(
        session_id=session_id, turn_id=turn_id, tenant_id=tenant_id, user_id=user_id,
        document_id=document_id, document_version=document_version,
        deck=render_deck(chunk_set.chunks),
        current_draft=render_for_reader(draft) if draft else None,
        draft_outline=artifact_outline(draft) if draft else [],
    )
    yield {
        "type": "start", "turn_id": turn_id,
        "source_ids": {sid: chunk.chunk_id for sid, chunk in source_ids(chunk_set.chunks).items()},
    }

    request = Command(resume=resume) if resume is not None else {"messages": [HumanMessage(message)]}
    text_parts: list[str] = []
    results: list[dict[str, Any]] = []
    drafts = FinalDrafts()
    usage: Counter = Counter()
    model_calls = 0
    first_token: float | None = None
    last_step: object = None
    step_start = 0
    tools_used = False
    try:
        async for chunk in agent.astream(
            request, thread_config(session_id), context=context,
            stream_mode=["messages", "updates", "custom"], version="v2", durability="exit",
        ):
            kind, data = chunk["type"], chunk["data"]
            if kind == "messages":
                piece, metadata = data
                if metadata.get("langgraph_node") != "model":
                    continue
                text = _chunk_text(piece)
                if not text:
                    continue
                step = metadata.get("langgraph_step")
                if step != last_step:
                    step_start = len(text_parts)
                if last_step is not None and step != last_step and text_parts:
                    text_parts.append("\n\n")
                    yield {"type": "token", "text": "\n\n"}
                last_step = step
                if first_token is None:
                    first_token = time.perf_counter() - started
                text_parts.append(text)
                yield {"type": "token", "text": text}
            elif kind == "custom":
                if data.get("type") == "retract":
                    removed = "".join(text_parts[step_start:])
                    del text_parts[step_start:]
                    last_step = object() if text_parts else None
                    yield {"type": "retract", "chars": len(removed)}
                    continue
                if data.get("type") == "artifact":
                    results.append(data["result"])
                    drafts.add(data)
                    continue
                yield data
            elif kind == "updates":
                if "__interrupt__" in data:
                    questions = [item.value for item in data["__interrupt__"]]
                    for item in drafts.items:
                        yield item
                    ledger.append_event(
                        **owner, role="assistant", event_type="agent_interrupt",
                        content=str(questions[0].get("question", "")) if questions else "",
                        payload={"turn_id": turn_id, "questions": questions,
                                 "results": drafts.results(), "all_results": results,
                                 "message": "".join(text_parts).strip()},
                    )
                    for question in questions:
                        yield {"type": "interrupt", "turn_id": turn_id, **question}
                    return
                if any(node.startswith("ModelCallLimitMiddleware") and isinstance(value, dict)
                       and value.get("messages") for node, value in data.items()):
                    note = ("\n\n" if text_parts else "") + LIMIT_REACHED
                    text_parts.append(note)
                    yield {"type": "token", "text": note}
                update = data.get("model") or {}
                for item in update.get("messages", []) if isinstance(update, dict) else []:
                    model_calls += 1
                    usage.update({
                        key: value for key, value in (getattr(item, "usage_metadata", None) or {}).items()
                        if isinstance(value, int)
                    })
                    for call in getattr(item, "tool_calls", None) or []:
                        tools_used = True
                        yield {"type": "step", "tool": call["name"], "stage": "start",
                               "detail": _TOOL_LABELS.get(call["name"], call["name"])}
    except asyncio.CancelledError:
        ledger.finish_turn(**owner, turn_id=turn_id, succeeded=False,
                           content="Stopped by the user.", payload={"kind": "agent_cancelled"})
        raise
    except Exception as error:
        logger.exception("Agent turn failed")
        busy = "503" in str(error) or "overloaded" in str(error).lower() or "UNAVAILABLE" in str(error)
        text = ("The AI model is busy right now. Nothing was lost; try again in a moment."
                if busy else "Something went wrong while answering. Try again.")
        ledger.finish_turn(**owner, turn_id=turn_id, succeeded=False, content=text,
                           payload={"kind": "agent_failed", "error": type(error).__name__,
                                    "results": drafts.results(), "all_results": results})
        for item in drafts.items:
            yield item
        yield {"type": "error", "message": text, "retryable": True}
        return

    for item in drafts.items:
        yield item
    answer = "".join(text_parts).strip()
    verification = None
    cited_only = tools_used or bool(results)
    if answer and (CITATION.search(answer) if cited_only else needs_check(answer)):
        yield {"type": "step", "tool": "answer_check", "stage": "checking",
               "detail": "Checking this answer against the deck"}
        try:
            verification = await check_answer(
                text=answer, chunks=chunk_set.chunks, verifier=services.verifier,
                document_id=document_id, document_version=document_version,
                user_request=message or "", cited_only=cited_only,
            )
        except Exception:
            logger.warning("Answer check failed", exc_info=True)
            verification = {"passed": None, "issues": ["The answer could not be checked right now."]}
        if verification is not None:
            yield {"type": "verification", **verification}
    elapsed = time.perf_counter() - started
    summary = {
        "latency_ms": round(elapsed * 1000),
        "first_token_ms": round(first_token * 1000) if first_token is not None else None,
        "model_calls": model_calls,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "tools": [result.get("step_id") for result in results],
    }
    ledger.finish_turn(
        **owner, turn_id=turn_id, succeeded=True, content=answer or "(done)",
        payload={"kind": "agent_result", "message": answer, "results": drafts.results(),
                 "all_results": results, "verification": verification, "usage": summary},
    )
    yield {"type": "done", **summary}
