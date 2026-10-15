"""Authorization-scope checks shared by retrieval implementations."""

from __future__ import annotations

from backend.app.chunking import ChunkSet


def validate_scope(
    chunk_set: ChunkSet, *, document_id: str, document_version: str
) -> None:
    if document_id != chunk_set.document_id:
        raise PermissionError("retrieval document is outside the authorized scope")
    if document_version != chunk_set.document_version:
        raise PermissionError("retrieval document version is outside the authorized scope")
