"""Atomic artifact, revision, and citation persistence."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping

import psycopg
from psycopg.types.json import Jsonb

from backend.app.generation import GeneratedArtifact

from .conversations import SessionAccessError, TurnConflictError
from .postgres import pooled_connection


class ArtifactRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def save_initial(
        self,
        *,
        artifact: GeneratedArtifact,
        session_id: str,
        turn_id: str,
        tenant_id: str,
        user_id: str,
        operation_step_id: str | None = None,
        operation_result: Mapping[str, Any] | None = None,
    ) -> None:
        revision_id = f"{artifact.artifact_id}:v{artifact.version}"
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """SELECT t.status, t.execution_plan FROM conversation_turns t
                   JOIN conversation_sessions s USING (session_id)
                   WHERE t.session_id = %s AND t.turn_id = %s
                     AND s.tenant_id = %s AND s.user_id = %s
                   FOR UPDATE OF t""",
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("turn not found for this tenant and user")
            if owned[1] is None:
                raise TurnConflictError("artifact cannot be stored before its plan")
            if not any(
                step.get("capability") == "generate"
                for step in owned[1].get("steps", [])
            ):
                raise TurnConflictError("artifact turn has no accepted generation step")
            connection.execute(
                """INSERT INTO artifacts
                   (artifact_id, session_id, document_version_id, output_specification)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (artifact_id) DO NOTHING""",
                (
                    artifact.artifact_id, session_id,
                    artifact.request.document_version,
                    Jsonb(asdict(artifact.request.output)),
                ),
            )
            inserted = connection.execute(
                """INSERT INTO artifact_revisions
                   (revision_id, artifact_id, version, content, model_id, prompt_version)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (artifact_id, version) DO NOTHING
                   RETURNING revision_id""",
                (
                    revision_id, artifact.artifact_id, artifact.version,
                    Jsonb(asdict(artifact)), artifact.model_id, artifact.prompt_version,
                ),
            ).fetchone()
            if inserted is None:
                if operation_step_id and operation_result:
                    _complete_operation(
                        connection, session_id=session_id, turn_id=turn_id,
                        step_id=operation_step_id, capability="generate",
                        result=operation_result,
                    )
                return
            for block in artifact.blocks:
                for unit in block.units:
                    for support in unit.support:
                        for evidence_id in support.evidence_ids:
                            connection.execute(
                                """INSERT INTO citations
                                   (citation_id, revision_id, unit_id, chunk_id, evidence_id)
                                   VALUES (%s, %s, %s, %s, %s)""",
                                (
                                    f"{revision_id}:{unit.unit_id}:{support.chunk_id}:{evidence_id}",
                                    revision_id, unit.unit_id, support.chunk_id, evidence_id,
                                ),
                            )
            connection.execute(
                """UPDATE conversation_sessions
                   SET active_artifact_id = %s, active_artifact_version = %s,
                       updated_at = now()
                   WHERE session_id = %s""",
                (artifact.artifact_id, artifact.version, session_id),
            )
            if operation_step_id and operation_result:
                _complete_operation(
                    connection, session_id=session_id, turn_id=turn_id,
                    step_id=operation_step_id, capability="generate",
                    result=operation_result,
                )

    def load_revision(
        self, *, artifact_id: str, version: int, tenant_id: str, user_id: str
    ) -> dict:
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """SELECT r.content FROM artifact_revisions r
                   JOIN artifacts a USING (artifact_id)
                   JOIN conversation_sessions s USING (session_id)
                   WHERE r.artifact_id = %s AND r.version = %s
                     AND s.tenant_id = %s AND s.user_id = %s""",
                (artifact_id, version, tenant_id, user_id),
            ).fetchone()
        if row is None:
            raise SessionAccessError("artifact revision not found for this principal")
        return row[0]

    def load_latest_for_session(
        self, *, session_id: str, tenant_id: str, user_id: str
    ) -> dict | None:
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """SELECT r.content FROM artifact_revisions r
                   JOIN artifacts a USING (artifact_id)
                   JOIN conversation_sessions s USING (session_id)
                   WHERE a.session_id = %s AND s.tenant_id = %s AND s.user_id = %s
                   ORDER BY r.created_at DESC, r.version DESC LIMIT 1""",
                (session_id, tenant_id, user_id),
            ).fetchone()
        return None if row is None else row[0]

    def save_revision(
        self,
        *,
        artifact: GeneratedArtifact,
        base_version: int,
        session_id: str,
        turn_id: str,
        tenant_id: str,
        user_id: str,
        edit_instruction: str,
        change_reason: str,
        operation_step_id: str | None = None,
        operation_result: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically append one edit revision with optimistic version checking."""
        if artifact.version != base_version + 1:
            raise ValueError("edited artifact version must immediately follow base version")
        if not edit_instruction.strip() or not change_reason.strip():
            raise ValueError("edit instruction and change reason are required")
        revision_id = f"{artifact.artifact_id}:v{artifact.version}"
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """SELECT t.execution_plan FROM conversation_turns t
                   JOIN conversation_sessions s USING (session_id)
                   JOIN artifacts a ON a.session_id = s.session_id
                   WHERE t.session_id = %s AND t.turn_id = %s
                     AND a.artifact_id = %s
                     AND s.tenant_id = %s AND s.user_id = %s
                   FOR UPDATE OF a, t""",
                (session_id, turn_id, artifact.artifact_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("artifact or turn not found for this principal")
            if owned[0] is None or not any(
                step.get("capability") == "edit" for step in owned[0].get("steps", [])
            ):
                raise TurnConflictError("artifact revision requires an accepted edit step")
            current = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM artifact_revisions WHERE artifact_id = %s",
                (artifact.artifact_id,),
            ).fetchone()[0]
            if current != base_version:
                raise TurnConflictError(
                    f"artifact version conflict: expected {base_version}, current {current}"
                )
            parent_id = f"{artifact.artifact_id}:v{base_version}"
            connection.execute(
                """INSERT INTO artifact_revisions
                   (revision_id, artifact_id, version, parent_revision_id, content,
                    edit_instruction, change_reason, model_id, prompt_version)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    revision_id, artifact.artifact_id, artifact.version, parent_id,
                    Jsonb(asdict(artifact)), edit_instruction, change_reason,
                    artifact.model_id, artifact.prompt_version,
                ),
            )
            self._insert_citations(connection, artifact=artifact, revision_id=revision_id)
            connection.execute(
                """UPDATE conversation_sessions
                   SET active_artifact_id = %s, active_artifact_version = %s,
                       updated_at = now()
                   WHERE session_id = %s""",
                (artifact.artifact_id, artifact.version, session_id),
            )
            if operation_step_id and operation_result:
                _complete_operation(
                    connection, session_id=session_id, turn_id=turn_id,
                    step_id=operation_step_id, capability="edit",
                    result=operation_result,
                )

    def load_active(
        self, *, session_id: str, tenant_id: str, user_id: str
    ) -> dict | None:
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """SELECT r.content FROM conversation_sessions s
                   JOIN artifact_revisions r
                     ON r.artifact_id = s.active_artifact_id
                    AND r.version = s.active_artifact_version
                   WHERE s.session_id = %s AND s.tenant_id = %s AND s.user_id = %s""",
                (session_id, tenant_id, user_id),
            ).fetchone()
        return None if row is None else row[0]

    def move_history(
        self, *, direction: str, session_id: str, turn_id: str,
        tenant_id: str, user_id: str, operation_step_id: str | None = None,
    ) -> dict:
        if direction not in {"undo", "redo"}:
            raise ValueError("history direction must be undo or redo")
        delta = -1 if direction == "undo" else 1
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """SELECT s.active_artifact_id, s.active_artifact_version,
                          t.execution_plan
                   FROM conversation_sessions s
                   JOIN conversation_turns t USING (session_id)
                   WHERE s.session_id = %s AND t.turn_id = %s
                     AND s.tenant_id = %s AND s.user_id = %s
                   FOR UPDATE OF s, t""",
                (session_id, turn_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                raise SessionAccessError("session or turn not found for this principal")
            artifact_id, version, plan = row
            if artifact_id is None or version is None:
                raise TurnConflictError("history operation requires an active artifact")
            if plan is None or not any(
                step.get("capability") == direction for step in plan.get("steps", [])
            ):
                raise TurnConflictError(f"accepted plan does not authorize {direction}")
            target_version = version + delta
            target = connection.execute(
                """SELECT content FROM artifact_revisions
                   WHERE artifact_id = %s AND version = %s""",
                (artifact_id, target_version),
            ).fetchone()
            if target is None:
                raise TurnConflictError(f"cannot {direction}: no adjacent revision")
            connection.execute(
                """UPDATE conversation_sessions SET active_artifact_version = %s,
                   updated_at = now() WHERE session_id = %s""",
                (target_version, session_id),
            )
            if operation_step_id:
                _complete_operation(
                    connection, session_id=session_id, turn_id=turn_id,
                    step_id=operation_step_id, capability=direction,
                    result={
                        "step_id": operation_step_id,
                        "status": "succeeded",
                        "value": {
                            "kind": direction, "artifact": target[0],
                            "reason": f"Moved to artifact revision {target_version}.",
                        },
                    },
                )
        return target[0]

    @staticmethod
    def _insert_citations(connection, *, artifact: GeneratedArtifact, revision_id: str) -> None:
        for block in artifact.blocks:
            for unit in block.units:
                for support in unit.support:
                    for evidence_id in support.evidence_ids:
                        connection.execute(
                            """INSERT INTO citations
                               (citation_id, revision_id, unit_id, chunk_id, evidence_id)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (
                                f"{revision_id}:{unit.unit_id}:{support.chunk_id}:{evidence_id}",
                                revision_id, unit.unit_id, support.chunk_id, evidence_id,
                            ),
                            )


def _complete_operation(
    connection, *, session_id: str, turn_id: str, step_id: str,
    capability: str, result: Mapping[str, Any],
) -> None:
    candidate = json.loads(json.dumps(dict(result), ensure_ascii=False))
    row = connection.execute(
        """
        UPDATE conversation_operations
        SET status = 'completed', result = %s, completed_at = now(), updated_at = now()
        WHERE session_id = %s AND turn_id = %s AND step_id = %s
          AND capability = %s AND status = 'running'
        RETURNING step_id
        """,
        (Jsonb(candidate), session_id, turn_id, step_id, capability),
    ).fetchone()
    if row is None:
        existing = connection.execute(
            """SELECT status, result, capability FROM conversation_operations
               WHERE session_id = %s AND turn_id = %s AND step_id = %s""",
            (session_id, turn_id, step_id),
        ).fetchone()
        if existing is None:
            raise TurnConflictError("operation must be claimed before mutation")
        if existing != ("completed", candidate, capability):
            raise TurnConflictError("operation already has a different mutation result")
