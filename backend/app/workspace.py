"""Single-process document workspace for the first browser integration slice."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from backend.app.chunking import ChunkSet, build_chunks
from backend.app.domain import ExtractedDocument
from backend.app.ingestion import extract_upload
from backend.app.retrieval import LexicalIndex, SearchHit
from backend.app.storage.assets import AssetStore


@dataclass(frozen=True, slots=True)
class WorkspaceDocument:
    document: ExtractedDocument
    chunk_set: ChunkSet


class InMemoryDocumentWorkspace:
    """Development store; explicit replacement target for PostgreSQL persistence."""

    def __init__(self) -> None:
        self._documents: dict[str, WorkspaceDocument] = {}
        self._lock = RLock()

    def ingest(self, payload: bytes, *, filename: str) -> WorkspaceDocument:
        return self.store(self.extract(payload, filename=filename))

    def extract(
        self, payload: bytes, *, filename: str, asset_store: AssetStore | None = None
    ) -> ExtractedDocument:
        """Extract only. Kept separate from `store` so pictures can be described
        between the two steps."""
        return extract_upload(payload, filename=filename, asset_store=asset_store)

    def store(
        self, document: ExtractedDocument, *, visual_evidence: tuple = ()
    ) -> WorkspaceDocument:
        stored = WorkspaceDocument(
            document, build_chunks(document, visual_evidence=visual_evidence)
        )
        with self._lock:
            self._documents[document.document_id] = stored
        return stored

    def get(self, document_id: str) -> WorkspaceDocument:
        with self._lock:
            stored = self._documents.get(document_id)
        if stored is None:
            raise KeyError("document is not present in this process workspace")
        return stored

    def retrieve(self, document_id: str, *, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        stored = self.get(document_id)
        return LexicalIndex(stored.chunk_set).search(
            query,
            document_id=stored.document.document_id,
            document_version=stored.document.document_version,
            top_k=top_k,
        )
