"""What the agent's tools receive: the run's identity, and the services they use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentContext:
    session_id: str
    turn_id: str
    tenant_id: str
    user_id: str
    document_id: str
    document_version: str
    deck: str = ""
    current_draft: str | None = None
    draft_outline: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentServices:
    """The existing SARAL components the tools wrap."""

    document_repository: Any
    artifact_repository: Any
    ledger: Any
    embedder: Any
    generator: Any
    verifier: Any
    editor: Any
