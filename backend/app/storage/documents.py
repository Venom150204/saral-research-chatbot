"""Tenant-scoped persistence for extracted documents and their retrieval index."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import psycopg
from psycopg.types.json import Jsonb

from backend.app.chunking.models import Chunk, ChunkConfig, ChunkSet, EvidenceReference
from backend.app.domain import ExtractedDocument

from .conversations import SessionAccessError
from .postgres import pooled_connection


class DocumentRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("database_url must be a PostgreSQL connection URL")
        self._database_url = database_url

    def save_index(
        self,
        *,
        document: ExtractedDocument,
        chunks: ChunkSet,
        vectors: Sequence[Sequence[float]],
        embedding_model_id: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        visual_evidence: Sequence[Any] = (),
    ) -> None:
        if len(vectors) != len(chunks.chunks):
            raise ValueError("each persisted chunk requires one embedding")
        if any(len(vector) != 768 for vector in vectors):
            raise ValueError("persisted embeddings must use the schema's 768 dimensions")
        if chunks.document_version != document.document_version:
            raise ValueError("chunk and document versions must match")
        if any(
            item.document_version != document.document_version for item in visual_evidence
        ):
            raise ValueError("visual evidence belongs to another document version")
        index_version = chunks.index_version
        with pooled_connection(self._database_url) as connection:
            owned = connection.execute(
                """SELECT session_id FROM conversation_sessions
                   WHERE session_id = %s AND tenant_id = %s AND user_id = %s
                   FOR UPDATE""",
                (session_id, tenant_id, user_id),
            ).fetchone()
            if owned is None:
                raise SessionAccessError("session not found for this tenant and user")
            connection.execute(
                """INSERT INTO documents
                   (document_id, tenant_id, owner_user_id, source_kind, display_name)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (document_id) DO NOTHING""",
                (
                    document.document_id, tenant_id, user_id,
                    _stored_kind(document.media_type),
                    document.filename,
                ),
            )
            owner = connection.execute(
                "SELECT tenant_id, owner_user_id FROM documents WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()
            if owner != (tenant_id, user_id):
                raise SessionAccessError("document identity belongs to another principal")
            connection.execute(
                """INSERT INTO document_versions
                   (document_version_id, document_id, content_sha256, parser_version)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (document_version_id) DO NOTHING""",
                (
                    document.document_version, document.document_id,
                    document.content_hash, f"{document.parser_name}:{document.parser_version}",
                ),
            )
            already_indexed = connection.execute(
                """SELECT 1 FROM chunks
                   WHERE document_version_id = %s AND index_version = %s LIMIT 1""",
                (document.document_version, index_version),
            ).fetchone() is not None
            for block in () if already_indexed else document.blocks:
                connection.execute(
                    """INSERT INTO source_blocks
                       (block_id, document_version_id, index_version, parent_id, ordinal,
                        block_type, original_text, location, hierarchy_level)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (document_version_id, index_version, block_id)
                       DO NOTHING""",
                    (
                        block.block_id, document.document_version, index_version,
                        block.parent_id, block.ordinal, block.block_type, block.text,
                        Jsonb(asdict(block.location)), block.hierarchy_level,
                    ),
                )
            for item in () if already_indexed else visual_evidence:
                connection.execute(
                    """INSERT INTO derived_visual_evidence
                       (evidence_id, document_version_id, visual_id, asset_sha256,
                        slide_number, shape_id, model_id, prompt_version, description)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (document_version_id, evidence_id) DO NOTHING""",
                    (
                        item.evidence_id, document.document_version, item.visual_id,
                        item.asset_hash, item.slide_number, item.shape_id,
                        item.model_id, item.prompt_version,
                        Jsonb({
                            "literal_description": item.literal_description,
                            "visible_text": list(item.visible_text),
                            "entities": list(item.entities),
                            "relationships": list(item.relationships),
                            "uncertainty": item.uncertainty,
                        }),
                    ),
                )
            for chunk, vector in () if already_indexed else zip(chunks.chunks, vectors, strict=True):
                metadata = {
                    "parent_id": chunk.parent_id,
                    "chunk_type": chunk.chunk_type,
                    "evidence_kind": chunk.evidence_kind,
                    "evidence": [asdict(item) for item in chunk.evidence],
                    "config_version": chunk.config_version,
                    "chunking_strategy": chunk.chunking_strategy,
                    "supports_quantitative_claims": chunk.supports_quantitative_claims,
                    "oversized": chunk.oversized,
                    "target_chars": chunks.config.target_chars,
                    "filter_repeated_slide_boilerplate": chunks.config.filter_repeated_slide_boilerplate,
                    "table_layout": chunks.config.table_layout,
                    "unit": chunks.config.unit,
                }
                vector_text = "[" + ",".join(str(float(value)) for value in vector) + "]"
                connection.execute(
                    """INSERT INTO chunks
                       (chunk_id, document_version_id, index_version, ordinal,
                        original_text, search_text, embedding, metadata,
                        embedding_model_id, embedded_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, now())
                       ON CONFLICT (chunk_id) DO UPDATE SET
                         embedding = EXCLUDED.embedding,
                         embedding_model_id = EXCLUDED.embedding_model_id,
                         embedded_at = EXCLUDED.embedded_at""",
                    (
                        chunk.chunk_id, document.document_version, index_version,
                        chunk.ordinal, chunk.text, chunk.search_text, vector_text,
                        Jsonb(metadata), embedding_model_id,
                    ),
                )
            connection.execute(
                """UPDATE document_versions SET active_index_version = %s
                   WHERE document_version_id = %s""",
                (index_version, document.document_version),
            )
            connection.execute(
                """UPDATE conversation_sessions SET active_document_version_id = %s,
                   updated_at = now() WHERE session_id = %s""",
                (document.document_version, session_id),
            )

    def display_name(self, *, document_id: str, tenant_id: str, user_id: str) -> str | None:
        """The uploaded file's name, for the owner only."""
        with pooled_connection(self._database_url) as connection:
            row = connection.execute(
                """SELECT display_name FROM documents
                   WHERE document_id = %s AND tenant_id = %s AND owner_user_id = %s""",
                (document_id, tenant_id, user_id),
            ).fetchone()
        return row[0] if row else None

    def load_chunk_set(
        self,
        *,
        document_version: str,
        tenant_id: str,
        user_id: str,
    ) -> tuple[ChunkSet, tuple[tuple[float, ...], ...], str]:
        with pooled_connection(self._database_url) as connection:
            owner = connection.execute(
                """SELECT d.document_id, v.active_index_version
                   FROM document_versions v
                   JOIN documents d USING (document_id)
                   WHERE v.document_version_id = %s
                     AND d.tenant_id = %s AND d.owner_user_id = %s""",
                (document_version, tenant_id, user_id),
            ).fetchone()
            if owner is None:
                raise SessionAccessError("document version not found for this principal")
            index_version = owner[1] or "legacy"
            rows = connection.execute(
                """SELECT chunk_id, ordinal, original_text, search_text, metadata,
                          embedding::text, embedding_model_id
                   FROM chunks
                   WHERE document_version_id = %s AND index_version = %s
                   ORDER BY ordinal""",
                (document_version, index_version),
            ).fetchall()
        if not rows:
            raise LookupError("document version has no persisted chunks")
        chunks = []
        vectors = []
        model_ids = set()
        for row in rows:
            metadata = row[4]
            evidence = tuple(EvidenceReference(**item) for item in metadata["evidence"])
            chunks.append(Chunk(
                chunk_id=row[0], document_id=owner[0], document_version=document_version,
                parent_id=metadata["parent_id"], ordinal=row[1],
                chunk_type=metadata["chunk_type"], evidence_kind=metadata["evidence_kind"],
                text=row[2], search_text=row[3], evidence=evidence,
                config_version=metadata["config_version"],
                chunking_strategy=metadata["chunking_strategy"],
                supports_quantitative_claims=metadata["supports_quantitative_claims"],
                oversized=metadata["oversized"],
            ))
            vectors.append(tuple(float(value) for value in row[5].strip("[]").split(",")))
            model_ids.add(row[6])
        if len(model_ids) != 1 or None in model_ids:
            raise ValueError("persisted chunk embeddings do not share one model")
        config = ChunkConfig(
            version=chunks[0].config_version,
            summary_strategy=chunks[0].chunking_strategy,
            target_chars=rows[0][4]["target_chars"],
            filter_repeated_slide_boilerplate=rows[0][4]["filter_repeated_slide_boilerplate"],
            table_layout=rows[0][4].get("table_layout", "row"),
            unit=rows[0][4].get("unit", "slide"),
        )
        return (
            ChunkSet(owner[0], document_version, config, tuple(chunks), index_version),
            tuple(vectors),
            model_ids.pop(),
        )


def _stored_kind(media_type: str) -> str:
    """The uploaded format, as the documents table records it."""
    return {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/pdf": "pdf",
        "application/x-tex": "latex",
    }.get(media_type, "summary")
