"""SARAL's conversational agent: one model with guarded tools, on LangGraph."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware, ModelRequest,
    ModelRetryMiddleware, SummarizationMiddleware, ToolCallLimitMiddleware, dynamic_prompt,
)

from backend.app.prompts import AGENT_PROMPT, render_prompt
from backend.app.prompts.agent import SUMMARY_PROMPT
from backend.app.provider_retry import is_transient

from .context import AgentContext, AgentServices
from .tools import build_tools

PROMPT_VERSION = "saral-agent-v6-safe-accessible"


@dataclass(frozen=True)
class AgentLimits:
    """Per-turn bounds. A deliverable turn needs two model calls (decide, report);
    a question or a compound request a few more."""

    model_calls: int = 6
    tool_calls: int = 5
    deliverables: int = 2
    edits: int = 3
    questions: int = 1
    summarize_after_tokens: int = 8_000
    keep_recent_messages: int = 12


class EmptyModelReply(RuntimeError):
    """The model answered with neither text nor a tool call, or wrote a tool call out
    as text instead of making it."""


_LEAKED_CALL = re.compile(r"call:\s*default_api:|default_api\.\w+\(|```tool_code|<\|?tool_call")


class RejectEmptyReplies(AgentMiddleware):
    """Turn an unusable model reply into an error, so the retry middleware around it asks
    again (and then the fallback model)."""

    @staticmethod
    def _check(request, response):
        message = response.result[-1] if response.result else None
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else "".join(
            part.get("text", "") for part in content if isinstance(part, dict))
        tool_calls = getattr(message, "tool_calls", None)
        if not text.strip() and not tool_calls:
            raise EmptyModelReply("the model returned an empty reply")
        if _LEAKED_CALL.search(text):
            writer = getattr(request.runtime, "stream_writer", None)
            if writer is not None:
                writer({"type": "retract"})
            raise EmptyModelReply("the model wrote a tool call as text")
        return response

    def wrap_model_call(self, request, handler):
        return self._check(request, handler(request))

    async def awrap_model_call(self, request, handler):
        return self._check(request, await handler(request))


def _retryable(error: BaseException) -> bool:
    return isinstance(error, EmptyModelReply) or is_transient(error) or (
        error.__cause__ is not None and is_transient(error.__cause__))


@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    context: AgentContext = request.runtime.context
    return render_prompt(
        AGENT_PROMPT,
        prompt_name="agent",
        DECK=context.deck or "(no deck uploaded)",
        CURRENT_DRAFT=context.current_draft or "None yet.",
        DRAFT_OUTLINE=json.dumps(context.draft_outline, ensure_ascii=False) if context.draft_outline else "None yet.",
    )


def build_agent(
    *,
    services: AgentServices,
    model: Any,
    fallback_model: Any | None = None,
    summary_model: Any | None = None,
    checkpointer: Any | None = None,
    limits: AgentLimits = AgentLimits(),
):
    middleware: list[Any] = [_system_prompt]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))
    middleware += [
        ModelRetryMiddleware(max_retries=2, retry_on=_retryable, on_failure="error"),
        RejectEmptyReplies(),
        ModelCallLimitMiddleware(run_limit=limits.model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=limits.tool_calls),
        ToolCallLimitMiddleware(tool_name="write_deliverable", run_limit=limits.deliverables),
        ToolCallLimitMiddleware(tool_name="edit_draft", run_limit=limits.edits),
        ToolCallLimitMiddleware(tool_name="ask_user", run_limit=limits.questions),
        SummarizationMiddleware(
            summary_model or model,
            trigger=("tokens", limits.summarize_after_tokens),
            keep=("messages", limits.keep_recent_messages),
            summary_prompt=SUMMARY_PROMPT,
        ),
    ]
    return create_agent(
        model,
        build_tools(services),
        middleware=middleware,
        context_schema=AgentContext,
        checkpointer=checkpointer,
        name="saral",
    )
