"""Tenant-scoped PostgreSQL repository for authoritative conversation history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Literal, Mapping

import psycopg
from psycopg.types.json import Jsonb
from .postgres import pooled_connection


EventRole = Literal["user", "assistant", "system", "tool"]
TurnStatus = Literal["running", "succeeded", "failed"]
_VALID_ROLES = {"user", "assistant", "system", "tool"}


class SessionAccessError(LookupError):
    """Raised when a session is absent or not owned by the supplied principal."""


class TurnConflictError(RuntimeError):
    """Raised when a turn ID is reused for a different request."""


@dataclass(frozen=True, slots=True)
class ConversationSession:
    session_id: str
    tenant_id: str
    user_id: str
    active_document_version_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StoredConversationEvent:
    event_id: int
    session_id: str
    sequence_no: int
    role: EventRole
    event_type: str
    content: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    session_id: str
    turn_id: str
    request_sha256: str
    status: TurnStatus
    user_event_id: int | None
    outcome_event_id: int | None
    execution_plan: dict[str, Any] | None
    planner_model_id: str | None
    planner_prompt_version: str | None
    planned_at: datetime | None
    created_at: datetime
    updated_at: datetime
    created: bool = False


@dataclass(frozen=True, slots=True)
class ConversationOperation:
    session_id: str
    turn_id: str
    step_id: str
    capability: str
    status: Literal["running", "completed"]
    attempt_count: int
    result: dict[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    updated_at: datetime


class ConversationRepository:
    """Open short transactions and enforce ownership in every session query."""

    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("database_url must be a PostgreSQL connection URL")
        self._database_url = database_url

    def create_session(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        active_document_version_id: str | None = None,
    ) -> ConversationSession:
        _require_text("session_id", session_id)
        _require_text("tenant_id", tenant_id)
        _require_text("user_id", user_id)
        with pooled_connection(self._database_url) as connection:
            connection.execute(
                """
                INSERT INTO conversation_sessions (
                    session_id, tenant_id, user_id, active_document_version_id
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, tenant_id, user_id, active_document_version_id),
            )
            row = _select_owned_session(
                connection,
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            if row is None:
                raise SessionAccessError("session not found for this tenant and user")
            return _session_from_row(row)

    def require_session(
        self, *, session_id: str, tenant_id: str, user_id: str
    ) -> ConversationSession:
        """Authorize a thread before its LangGraph checkpoint is accessed."""
        with pooled_connection(self._database_url) as connection:
            row = _select_owned_session(
                connection,
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        if row is None:
            raise SessionAccessError("session not found for this tenant and user")
        return _session_from_row(row)

    def append_event(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        role: EventRole,
        event_type: str,
        content: str,
        payload: Mapping[str, Any] | None = None,
    ) -> StoredConversationEvent:
        if role not in _VALID_ROLES:
            raise ValueError(f"unsupported conversation role: {role}")
        _require_text("event_type", event_type)
        _require_text("content", content)
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """
                SELECT session_id
                FROM conversation_sessions
                WHERE session_id = %s AND tenant_id = %s AND user_id = %s
                FOR UPDATE
                """,
                (session_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("session not found for this tenant and user")
            row = _insert_event(
                connection,
                session_id=session_id,
                role=role,
                event_type=event_type,
                content=content,
                payload=payload,
            )
            connection.execute(
                "UPDATE conversation_sessions SET updated_at = now() WHERE session_id = %s",
                (session_id,),
            )
        return _event_from_row(row)

    def begin_turn(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        request_sha256: str,
        message: str,
    ) -> tuple[ConversationTurn, StoredConversationEvent]:
        """Atomically register a turn and its single raw user event."""
        _require_text("turn_id", turn_id)
        if len(request_sha256) != 64:
            raise ValueError("request_sha256 must contain 64 hexadecimal characters")
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """
                SELECT session_id FROM conversation_sessions
                WHERE session_id = %s AND tenant_id = %s AND user_id = %s
                FOR UPDATE
                """,
                (session_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("session not found for this tenant and user")
            inserted = connection.execute(
                """
                INSERT INTO conversation_turns (
                    session_id, turn_id, request_sha256, status
                ) VALUES (%s, %s, %s, 'running')
                ON CONFLICT (session_id, turn_id) DO NOTHING
                RETURNING session_id
                """,
                (session_id, turn_id, request_sha256),
            ).fetchone()
            event = None
            if inserted is not None:
                event = _insert_event(
                    connection,
                    session_id=session_id,
                    role="user",
                    event_type="message",
                    content=message,
                    payload={"turn_id": turn_id},
                )
                connection.execute(
                    """
                    UPDATE conversation_turns
                    SET user_event_id = %s, updated_at = now()
                    WHERE session_id = %s AND turn_id = %s
                    """,
                    (event[0], session_id, turn_id),
                )
            row = _select_turn(connection, session_id=session_id, turn_id=turn_id)
            if row[2] != request_sha256:
                raise TurnConflictError("turn ID is already bound to another request")
            if event is None:
                event = _select_event(connection, event_id=row[4])
        return _turn_from_row(row, created=inserted is not None), _event_from_row(event)

    def finish_turn(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        succeeded: bool,
        content: str,
        payload: Mapping[str, Any],
    ) -> tuple[ConversationTurn, StoredConversationEvent]:
        """Atomically append one outcome and close a running turn."""
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """
                SELECT t.session_id
                FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                FOR UPDATE OF t
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("turn not found for this tenant and user")
            current = _select_turn(connection, session_id=session_id, turn_id=turn_id)
            if current[3] != "running":
                raise TurnConflictError("turn already has a terminal outcome")
            status = "succeeded" if succeeded else "failed"
            event = _insert_event(
                connection,
                session_id=session_id,
                role="assistant" if succeeded else "system",
                event_type="workflow_action" if succeeded else "workflow_failed",
                content=content,
                payload=payload,
            )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = %s, outcome_event_id = %s, updated_at = now()
                WHERE session_id = %s AND turn_id = %s
                """,
                (status, event[0], session_id, turn_id),
            )
            row = _select_turn(connection, session_id=session_id, turn_id=turn_id)
        return _turn_from_row(row), _event_from_row(event)

    def save_turn_plan(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        plan: Mapping[str, Any],
        model_id: str,
        prompt_version: str,
    ) -> ConversationTurn:
        """Persist one immutable accepted plan on its authorized running turn."""
        _require_text("model_id", model_id)
        _require_text("prompt_version", prompt_version)
        candidate = dict(plan)
        if not candidate:
            raise ValueError("execution plan is required")
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """
                SELECT t.execution_plan, t.planner_model_id, t.planner_prompt_version
                FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                FOR UPDATE OF t
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                raise SessionAccessError("turn not found for this tenant and user")
            if row[0] is not None:
                if row != (candidate, model_id, prompt_version):
                    raise TurnConflictError("turn already has a different accepted plan")
            else:
                connection.execute(
                    """
                    UPDATE conversation_turns
                    SET execution_plan = %s, planner_model_id = %s,
                        planner_prompt_version = %s, planned_at = now(),
                        updated_at = now()
                    WHERE session_id = %s AND turn_id = %s AND status = 'running'
                    """,
                    (Jsonb(candidate), model_id, prompt_version, session_id, turn_id),
                )
            stored = _select_turn(connection, session_id=session_id, turn_id=turn_id)
        return _turn_from_row(stored)

    def get_turn(
        self, *, session_id: str, tenant_id: str, user_id: str, turn_id: str
    ) -> ConversationTurn:
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """
                SELECT t.session_id FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("turn not found for this tenant and user")
            row = _select_turn(connection, session_id=session_id, turn_id=turn_id)
        return _turn_from_row(row)

    def get_turn_outcome(
        self, *, session_id: str, tenant_id: str, user_id: str, turn_id: str
    ) -> StoredConversationEvent | None:
        """Return the authorized terminal event for idempotent API replay."""
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """
                SELECT e.event_id, e.session_id, e.sequence_no, e.role,
                       e.event_type, e.content, e.payload, e.created_at
                FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                JOIN conversation_events e ON e.event_id = t.outcome_event_id
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
        return None if row is None else _event_from_row(row)

    def get_turn_request(
        self, *, session_id: str, tenant_id: str, user_id: str, turn_id: str
    ) -> StoredConversationEvent:
        """Return the immutable authorized user event bound to a turn."""
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """
                SELECT e.event_id, e.session_id, e.sequence_no, e.role,
                       e.event_type, e.content, e.payload, e.created_at
                FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                JOIN conversation_events e ON e.event_id = t.user_event_id
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
        if row is None:
            raise SessionAccessError("turn request not found for this tenant and user")
        return _event_from_row(row)

    def claim_operation(
        self, *, session_id: str, tenant_id: str, user_id: str,
        turn_id: str, step_id: str, capability: str,
    ) -> ConversationOperation:
        """Create/reclaim one authorized operation without changing completed work."""
        _require_text("step_id", step_id)
        _require_text("capability", capability)
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """
                SELECT t.status FROM conversation_turns t
                JOIN conversation_sessions s USING (session_id)
                WHERE t.session_id = %s AND t.turn_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                FOR UPDATE OF t
                """,
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("turn not found for this tenant and user")
            row = connection.execute(
                """
                INSERT INTO conversation_operations (
                    session_id, turn_id, step_id, capability, status
                ) VALUES (%s, %s, %s, %s, 'running')
                ON CONFLICT (session_id, turn_id, step_id) DO UPDATE
                SET attempt_count = CASE
                        WHEN conversation_operations.status = 'running'
                        THEN conversation_operations.attempt_count + 1
                        ELSE conversation_operations.attempt_count
                    END,
                    updated_at = now()
                RETURNING session_id, turn_id, step_id, capability, status,
                          attempt_count, result, started_at, completed_at, updated_at
                """,
                (session_id, turn_id, step_id, capability),
            ).fetchone()
            if row[3] != capability:
                raise TurnConflictError("operation step is bound to another capability")
        return _operation_from_row(row)

    def complete_operation(
        self, *, session_id: str, tenant_id: str, user_id: str,
        turn_id: str, step_id: str, capability: str, result: Mapping[str, Any],
    ) -> ConversationOperation:
        """Store one immutable typed result for graph resume and replay."""
        candidate = json.loads(json.dumps(dict(result), ensure_ascii=False))
        if not candidate:
            raise ValueError("operation result is required")
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """
                SELECT o.session_id, o.turn_id, o.step_id, o.capability, o.status,
                       o.attempt_count, o.result, o.started_at, o.completed_at, o.updated_at
                FROM conversation_operations o
                JOIN conversation_sessions s USING (session_id)
                WHERE o.session_id = %s AND o.turn_id = %s AND o.step_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                FOR UPDATE OF o
                """,
                (session_id, turn_id, step_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                raise SessionAccessError("operation not found for this tenant and user")
            if row[3] != capability:
                raise TurnConflictError("operation step is bound to another capability")
            if row[4] == "completed":
                if row[6] != candidate:
                    raise TurnConflictError("operation already has a different result")
                return _operation_from_row(row)
            row = connection.execute(
                """
                UPDATE conversation_operations
                SET status = 'completed', result = %s, completed_at = now(), updated_at = now()
                WHERE session_id = %s AND turn_id = %s AND step_id = %s
                RETURNING session_id, turn_id, step_id, capability, status,
                          attempt_count, result, started_at, completed_at, updated_at
                """,
                (Jsonb(candidate), session_id, turn_id, step_id),
            ).fetchone()
        return _operation_from_row(row)

    def get_operation(
        self, *, session_id: str, tenant_id: str, user_id: str,
        turn_id: str, step_id: str,
    ) -> ConversationOperation | None:
        self.require_session(
            session_id=session_id, tenant_id=tenant_id, user_id=user_id
        )
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """
                SELECT o.session_id, o.turn_id, o.step_id, o.capability, o.status,
                       o.attempt_count, o.result, o.started_at, o.completed_at, o.updated_at
                FROM conversation_operations o
                JOIN conversation_sessions s USING (session_id)
                WHERE o.session_id = %s AND o.turn_id = %s AND o.step_id = %s
                  AND s.tenant_id = %s AND s.user_id = %s
                """,
                (session_id, turn_id, step_id, tenant_id, user_id),
            ).fetchone()
        return None if row is None else _operation_from_row(row)

    def list_events(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_id: str,
        limit: int = 100,
        before_sequence: int | None = None,
        after_sequence: int | None = None,
    ) -> tuple[StoredConversationEvent, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("event limit must be between 1 and 500")
        with pooled_connection(self._database_url) as connection:
            owned = _select_owned_session(
                connection,
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            if owned is None:
                raise SessionAccessError("session not found for this tenant and user")
            rows = connection.execute(
                """
                SELECT event_id, session_id, sequence_no, role, event_type,
                       content, payload, created_at
                FROM conversation_events
                WHERE session_id = %s
                  AND (%s::bigint IS NULL OR sequence_no < %s::bigint)
                  AND (%s::bigint IS NULL OR sequence_no > %s::bigint)
                ORDER BY sequence_no DESC
                LIMIT %s
                """,
                (session_id, before_sequence, before_sequence,
                 after_sequence, after_sequence, limit),
            ).fetchall()
        return tuple(_event_from_row(row) for row in reversed(rows))


def _require_owned_session(connection, **owner) -> None:
    if _select_owned_session(connection, **owner) is None:
        raise SessionAccessError("session not found for this tenant and user")


def _select_owned_session(connection, *, session_id: str, tenant_id: str, user_id: str):
    return connection.execute(
        """
        SELECT session_id, tenant_id, user_id, active_document_version_id,
               created_at, updated_at
        FROM conversation_sessions
        WHERE session_id = %s AND tenant_id = %s AND user_id = %s
        """,
        (session_id, tenant_id, user_id),
    ).fetchone()


def _select_turn(connection, *, session_id: str, turn_id: str):
    return connection.execute(
        """
        SELECT session_id, turn_id, request_sha256, status, user_event_id,
               outcome_event_id, execution_plan, planner_model_id,
               planner_prompt_version, planned_at, created_at, updated_at
        FROM conversation_turns
        WHERE session_id = %s AND turn_id = %s
        """,
        (session_id, turn_id),
    ).fetchone()


def _select_event(connection, *, event_id: int):
    return connection.execute(
        """
        SELECT event_id, session_id, sequence_no, role, event_type,
               content, payload, created_at
        FROM conversation_events WHERE event_id = %s
        """,
        (event_id,),
    ).fetchone()


def _insert_event(
    connection,
    *,
    session_id: str,
    role: EventRole,
    event_type: str,
    content: str,
    payload: Mapping[str, Any] | None,
):
    next_sequence = connection.execute(
        """
        SELECT COALESCE(MAX(sequence_no), -1) + 1
        FROM conversation_events WHERE session_id = %s
        """,
        (session_id,),
    ).fetchone()[0]
    return connection.execute(
        """
        INSERT INTO conversation_events (
            session_id, sequence_no, role, event_type, content, payload
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING event_id, session_id, sequence_no, role, event_type,
                  content, payload, created_at
        """,
        (session_id, next_sequence, role, event_type, content, Jsonb(dict(payload or {}))),
    ).fetchone()


def _session_from_row(row) -> ConversationSession:
    return ConversationSession(*row)


def _event_from_row(row) -> StoredConversationEvent:
    return StoredConversationEvent(*row)


def _operation_from_row(row) -> ConversationOperation:
    return ConversationOperation(*row)


def _turn_from_row(row, *, created: bool = False) -> ConversationTurn:
    return ConversationTurn(*row, created=created)


def _require_text(field: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} is required")
