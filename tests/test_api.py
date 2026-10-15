"""The HTTP API and PostgreSQL storage: upload, sessions, ledger, versions, server start-up."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from backend.app.api import create_app
import os
from dataclasses import asdict, replace
from uuid import uuid4
from backend.app.storage import (
    ConversationRepository,
    SessionAccessError,
    TurnConflictError,
    apply_domain_migrations,
    DocumentRepository,
    ArtifactRepository,
)
from backend.app.embeddings import EmbeddingBatch
from backend.app.ingestion import extract_summary_bytes
from backend.app.chunking import build_chunks
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference,
)
from backend.app.agent.runner import AGENT_PLAN
from backend.app.workflow import StepResult
import asyncio
from uvicorn.config import Config


class ApiTests(unittest.TestCase):
    def test_uploads_summary_and_retrieves_scoped_evidence(self) -> None:
        with TestClient(create_app(use_environment_repository=False)) as client:
            upload = client.post(
                "/api/documents?filename=findings.md",
                content=b"# Results\n\nAttention replaces recurrence in the architecture.\n",
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(upload.status_code, 200)
            document = upload.json()
            retrieval = client.post(
                f"/api/documents/{document['document_id']}/retrieve",
                json={"query": "attention recurrence", "top_k": 3},
            )
            sources = client.get(f"/api/documents/{document['document_id']}/sources")
            missing_sources = client.get("/api/documents/missing/sources")

        self.assertEqual(document["persistence"], "process_memory")
        self.assertGreaterEqual(document["chunk_count"], 1)
        self.assertEqual(sources.status_code, 200)
        self.assertEqual(len(sources.json()["sources"]), document["chunk_count"])
        self.assertIn(
            "Attention replaces recurrence",
            " ".join(item["text"] for item in sources.json()["sources"]),
        )
        self.assertEqual(missing_sources.status_code, 404)
        self.assertEqual(retrieval.status_code, 200)
        self.assertIn("Attention replaces recurrence", retrieval.json()["hits"][0]["text"])
        self.assertTrue(retrieval.json()["hits"][0]["evidence"])

    def test_rejects_unsupported_upload_and_unknown_document(self) -> None:
        with TestClient(create_app(use_environment_repository=False)) as client:
            unsupported = client.post(
                "/api/documents?filename=paper.pdf", content=b"not a pdf"
            )
            missing = client.post(
                "/api/documents/missing/retrieve", json={"query": "anything"}
            )
        self.assertEqual(unsupported.status_code, 422)
        self.assertEqual(missing.status_code, 404)

    def test_each_role_can_have_its_own_model_and_defaults_to_the_shared_one(self) -> None:
        import os
        from unittest import mock

        roles = {
            "SARAL_TEXT_MODEL": "shared-model", "SARAL_WRITER_MODEL": "",
            "SARAL_VERIFIER_MODEL": "checker-model",
        }
        with mock.patch.dict(os.environ, roles):
            with TestClient(create_app(use_environment_repository=False)) as client:
                health = client.get("/health").json()
        self.assertEqual(
            (health["generation_model"], health["verifier_model"]),
            ("shared-model", "checker-model"),
        )

    def test_serves_the_built_ui_and_health(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            built = Path(folder)
            (built / "assets").mkdir()
            (built / "index.html").write_text('<title>SARAL</title><div id="root"></div>', encoding="utf-8")
            (built / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
            with TestClient(create_app(use_environment_repository=False, frontend_dir=built)) as client:
                page = client.get("/")
                script = client.get("/assets/index-abc.js")
                health = client.get("/health")
        self.assertEqual(page.status_code, 200)
        self.assertIn('<div id="root">', page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertEqual(script.status_code, 200)

    def test_the_database_url_falls_back_to_neons_variables(self) -> None:
        import os
        from unittest import mock

        from backend.app.storage import database_url_from_env

        clean = {name: "" for name in ("SARAL_DATABASE_URL", "DATABASE_URL_UNPOOLED", "DATABASE_URL")}
        with mock.patch.dict(os.environ, {**clean, "DATABASE_URL": "postgresql://pooled"}):
            self.assertEqual(database_url_from_env(), "postgresql://pooled")
        with mock.patch.dict(os.environ, {**clean, "DATABASE_URL": "postgresql://pooled",
                                          "DATABASE_URL_UNPOOLED": "postgresql://direct"}):
            self.assertEqual(database_url_from_env(), "postgresql://direct")
        with mock.patch.dict(os.environ, {**clean, "DATABASE_URL_UNPOOLED": "postgresql://direct",
                                          "SARAL_DATABASE_URL": "postgresql://explicit"}):
            self.assertEqual(database_url_from_env(), "postgresql://explicit")
        with mock.patch.dict(os.environ, clean):
            self.assertIsNone(database_url_from_env())

    def test_an_unbuilt_ui_says_how_to_build_it(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(use_environment_repository=False, frontend_dir=Path(folder))) as client:
                page = client.get("/")
                health = client.get("/health")
        self.assertEqual(page.status_code, 200)
        self.assertIn("npm run build", page.text)
        self.assertEqual(health.json()["surface"], "connected-development")
        self.assertFalse(health.json()["postgres_configured"])

    def test_durable_session_requires_postgres_configuration(self) -> None:
        with TestClient(create_app(use_environment_repository=False)) as client:
            response = client.post("/api/sessions")
        self.assertEqual(response.status_code, 503)
        self.assertIn("SARAL_DATABASE_URL", response.json()["detail"])


TEST_DATABASE_URL = os.getenv("SARAL_TEST_DATABASE_URL")

@unittest.skipUnless(TEST_DATABASE_URL, "SARAL_TEST_DATABASE_URL is not configured")
class PostgresIntegrationTests(unittest.TestCase):
    def test_operation_ledger_is_tenant_scoped_immutable_and_replayable(self) -> None:
        assert TEST_DATABASE_URL is not None
        apply_domain_migrations(TEST_DATABASE_URL)
        repository = ConversationRepository(TEST_DATABASE_URL)
        owner = {"tenant_id": "ledger-tenant", "user_id": "ledger-user"}
        session_id, turn_id = f"ledger-{uuid4()}", f"turn-{uuid4()}"
        repository.create_session(session_id=session_id, **owner)
        repository.begin_turn(
            session_id=session_id, turn_id=turn_id,
            request_sha256="b" * 64, message="Create a summary.", **owner,
        )
        first = repository.claim_operation(
            session_id=session_id, turn_id=turn_id, step_id="step-1",
            capability="generate", **owner,
        )
        self.assertEqual(first.status, "running")
        completed = repository.complete_operation(
            session_id=session_id, turn_id=turn_id, step_id="step-1",
            capability="generate",
            result={"step_id": "step-1", "status": "succeeded", "value": {"kind": "fixture"}},
            **owner,
        )
        replay = repository.claim_operation(
            session_id=session_id, turn_id=turn_id, step_id="step-1",
            capability="generate", **owner,
        )
        self.assertEqual(completed.result, replay.result)
        self.assertEqual(replay.status, "completed")
        self.assertEqual(replay.attempt_count, 1)
        with self.assertRaises(TurnConflictError):
            repository.claim_operation(
                session_id=session_id, turn_id=turn_id, step_id="step-1",
                capability="edit", **owner,
            )
        with self.assertRaises(SessionAccessError):
            repository.get_operation(
                session_id=session_id, turn_id=turn_id, step_id="step-1",
                tenant_id="other", user_id="other",
            )

    def test_edit_revision_is_atomic_and_rejects_stale_base(self) -> None:
        assert TEST_DATABASE_URL is not None
        apply_domain_migrations(TEST_DATABASE_URL)
        conversations = ConversationRepository(TEST_DATABASE_URL)
        documents = DocumentRepository(TEST_DATABASE_URL)
        artifacts = ArtifactRepository(TEST_DATABASE_URL)
        owner = {"tenant_id": "development-tenant", "user_id": "development-user"}
        session_id = f"edit-{uuid4()}"
        conversations.create_session(session_id=session_id, **owner)
        document = extract_summary_bytes(b"# Finding\n\nAttention replaces recurrence.\n", filename="edit.md")
        chunks = build_chunks(document)
        documents.save_index(
            document=document, chunks=chunks,
            vectors=tuple(tuple([1.0] * 768) for _ in chunks.chunks),
            embedding_model_id="fixture", session_id=session_id, **owner,
        )
        generate_turn = f"generate-{uuid4()}"
        conversations.begin_turn(
            session_id=session_id, turn_id=generate_turn, message="Create a card",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=generate_turn, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        chunk = chunks.chunks[0]
        support = (SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,)),)
        original = GeneratedArtifact(
            f"artifact-{uuid4()}", 1,
            GenerationRequest(document.document_id, document.document_version, OutputSpecification("card")),
            (ArtifactBlock("card", "card", (GroundedText("claim", "claim", "Attention replaces recurrence.", True, support),)),),
            "generator", "v1",
        )
        artifacts.save_initial(artifact=original, session_id=session_id, turn_id=generate_turn, **owner)
        edit_turn = f"edit-{uuid4()}"
        conversations.begin_turn(
            session_id=session_id, turn_id=edit_turn, message="Make claim clearer",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=edit_turn, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        edited_unit = replace(original.blocks[0].units[0], text="The architecture uses attention instead of recurrence.")
        edited = replace(
            original, version=2,
            blocks=(replace(original.blocks[0], units=(edited_unit,)),),
            model_id="editor", prompt_version="edit-v1",
        )
        edit_value = {
            "kind": "edited_artifact", "artifact": asdict(edited),
            "deltas": [], "reason": "Clarified wording",
        }
        conversations.claim_operation(
            session_id=session_id, turn_id=edit_turn, step_id="edit",
            capability="edit", **owner,
        )
        artifacts.save_revision(
            artifact=edited, base_version=1, session_id=session_id, turn_id=edit_turn,
            edit_instruction="Make claim clearer", change_reason="Clarified wording", **owner,
            operation_step_id="edit",
            operation_result=asdict(StepResult("edit", "succeeded", edit_value)),
        )
        self.assertEqual(conversations.get_operation(
            session_id=session_id, turn_id=edit_turn, step_id="edit", **owner,
        ).status, "completed")
        latest = artifacts.load_latest_for_session(session_id=session_id, **owner)
        self.assertEqual(latest["version"], 2)
        self.assertEqual(latest["blocks"][0]["units"][0]["text"], edited_unit.text)
        with self.assertRaisesRegex(TurnConflictError, "version conflict"):
            artifacts.save_revision(
                artifact=edited, base_version=1, session_id=session_id, turn_id=edit_turn,
                edit_instruction="Retry stale edit", change_reason="Should fail", **owner,
            )
        undo_turn = f"undo-{uuid4()}"
        conversations.begin_turn(
            session_id=session_id, turn_id=undo_turn, message="Undo",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=undo_turn, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        conversations.claim_operation(
            session_id=session_id, turn_id=undo_turn, step_id="undo",
            capability="undo", **owner,
        )
        undone = artifacts.move_history(
            direction="undo", session_id=session_id, turn_id=undo_turn,
            operation_step_id="undo", **owner,
        )
        self.assertEqual(conversations.get_operation(
            session_id=session_id, turn_id=undo_turn, step_id="undo", **owner,
        ).status, "completed")
        self.assertEqual(undone["version"], 1)
        self.assertEqual(artifacts.load_active(session_id=session_id, **owner)["version"], 1)
        lower_bound_turn = f"undo-boundary-{uuid4()}"
        conversations.begin_turn(
            session_id=session_id, turn_id=lower_bound_turn, message="Undo again",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=lower_bound_turn, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        with self.assertRaisesRegex(TurnConflictError, "no adjacent revision"):
            artifacts.move_history(
                direction="undo", session_id=session_id, turn_id=lower_bound_turn, **owner,
            )
        redo_turn = f"redo-{uuid4()}"
        conversations.begin_turn(
            session_id=session_id, turn_id=redo_turn, message="Redo",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=redo_turn, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        conversations.claim_operation(
            session_id=session_id, turn_id=redo_turn, step_id="redo",
            capability="redo", **owner,
        )
        redone = artifacts.move_history(
            direction="redo", session_id=session_id, turn_id=redo_turn,
            operation_step_id="redo", **owner,
        )
        self.assertEqual(conversations.get_operation(
            session_id=session_id, turn_id=redo_turn, step_id="redo", **owner,
        ).status, "completed")
        self.assertEqual(redone["version"], 2)

    def test_grounded_artifact_revision_and_citations_commit_after_generate_plan(self) -> None:
        assert TEST_DATABASE_URL is not None
        apply_domain_migrations(TEST_DATABASE_URL)
        conversations = ConversationRepository(TEST_DATABASE_URL)
        documents = DocumentRepository(TEST_DATABASE_URL)
        session_id, turn_id = f"artifact-{uuid4()}", f"turn-{uuid4()}"
        owner = {"tenant_id": "development-tenant", "user_id": "development-user"}
        conversations.create_session(session_id=session_id, **owner)
        document = extract_summary_bytes(
            b"# Finding\n\nAttention replaces recurrence.\n", filename="artifact.md"
        )
        chunks = build_chunks(document)
        vectors = tuple(tuple([1.0] * 768) for _ in chunks.chunks)
        documents.save_index(
            document=document, chunks=chunks, vectors=vectors,
            embedding_model_id="fixture-embedding", session_id=session_id, **owner,
        )
        conversations.begin_turn(
            session_id=session_id, turn_id=turn_id, message="Create an unusual evidence card.",
            request_sha256="a" * 64, **owner,
        )
        conversations.save_turn_plan(session_id=session_id, turn_id=turn_id, plan=AGENT_PLAN,
                                    model_id="fixture-agent", prompt_version="fixture", **owner)
        chunk = chunks.chunks[0]
        artifact = GeneratedArtifact(
            artifact_id=f"artifact-{uuid4()}", version=1,
            request=GenerationRequest(
                document.document_id, document.document_version,
                OutputSpecification(format="an unusual evidence card"),
            ),
            blocks=(ArtifactBlock(
                "card-1", "evidence_card",
                (GroundedText(
                    "claim-1", "claim", "Attention replaces recurrence.", True,
                    (SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,)),),
                ),),
            ),),
            model_id="fixture-generator", prompt_version="fixture-generation-v1",
        )
        artifacts = ArtifactRepository(TEST_DATABASE_URL)
        artifacts.save_initial(
            artifact=artifact, session_id=session_id, turn_id=turn_id, **owner
        )
        restored = ArtifactRepository(TEST_DATABASE_URL).load_revision(
            artifact_id=artifact.artifact_id, version=1, **owner
        )
        self.assertEqual(restored["blocks"][0]["block_type"], "evidence_card")
        self.assertEqual(
            restored["blocks"][0]["units"][0]["support"][0]["chunk_id"],
            chunk.chunk_id,
        )

    def test_durable_upload_restores_chunks_embeddings_and_active_document(self) -> None:
        assert TEST_DATABASE_URL is not None
        apply_domain_migrations(TEST_DATABASE_URL)
        conversation_repository = ConversationRepository(TEST_DATABASE_URL)
        document_repository = DocumentRepository(TEST_DATABASE_URL)
        with TestClient(create_app(
            repository=conversation_repository,
            document_repository=document_repository,
            embedder=_FakeEmbedder(),
        )) as client:
            session_id = client.post("/api/sessions").json()["session_id"]
            uploaded = client.post(
                f"/api/documents?filename=study.md&session_id={session_id}",
                content=b"# Finding\n\nAttention replaces recurrence.\n",
                headers={"Content-Type": "application/octet-stream"},
            )
        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()["persistence"], "postgresql_pgvector")

        reopened_documents = DocumentRepository(TEST_DATABASE_URL)
        chunk_set, vectors, model_id = reopened_documents.load_chunk_set(
            document_version=uploaded.json()["document_version"],
            tenant_id="development-tenant",
            user_id="development-user",
        )
        self.assertEqual(model_id, "fixture-embedding")
        self.assertEqual(len(vectors), len(chunk_set.chunks))
        self.assertEqual(len(vectors[0]), 768)
        with TestClient(create_app(
            repository=ConversationRepository(TEST_DATABASE_URL),
            document_repository=reopened_documents,
            embedder=_FakeEmbedder(),
        )) as client:
            resumed = client.get(f"/api/sessions/{session_id}")
            retrieved = client.post(
                f"/api/documents/{uploaded.json()['document_id']}/retrieve",
                json={
                    "query": "attention recurrence",
                    "document_version": uploaded.json()["document_version"],
                    "session_id": session_id,
                },
            )
        self.assertEqual(resumed.json()["active_document"]["document_id"], uploaded.json()["document_id"])
        self.assertEqual(resumed.json()["active_document"]["filename"], "study.md")
        self.assertEqual(resumed.json()["active_document"]["source_kind"], "summary")
        self.assertTrue(retrieved.json()["hits"])
        self.assertEqual(retrieved.json()["retrieval_mode"], "hybrid_rrf")
        with TestClient(create_app(
            repository=ConversationRepository(TEST_DATABASE_URL),
            document_repository=reopened_documents,
            embedder=_FakeEmbedder(),
        )) as client:
            attributed = client.post(
                f"/api/documents/{uploaded.json()['document_id']}/retrieve",
                json={
                    "query": "Attention is used in place of recurrence.",
                    "document_version": uploaded.json()["document_version"],
                    "mode": "attribution", "per_channel": 1,
                },
            )
        self.assertEqual(attributed.status_code, 200)
        self.assertEqual(attributed.json()["retrieval_mode"], "pooled_candidates")
        support = attributed.json()["hits"][0]
        self.assertEqual(support["text"], "Attention replaces recurrence.")
        self.assertEqual(support["chunk_type"], "paragraph")
        self.assertEqual(support["lexical_rank"], 1)

    def test_described_pictures_are_embedded_and_reingestion_keeps_the_older_index(self) -> None:
        assert TEST_DATABASE_URL is not None
        import psycopg
        from tests.decks import build_representative_visual_deck
        from backend.app.visuals import VisualDescription, VisualDescriptionResult

        class Unavailable:
            async def describe(self, request):
                raise RuntimeError("vision provider unavailable")

        class Describer:
            async def describe(self, request):
                return VisualDescriptionResult(
                    description=VisualDescription(
                        visual_id=request.visual_id, model_id="fixture-vlm",
                        prompt_version="fixture-prompt-v1",
                        literal_description="Control and treated panels side by side.",
                        visible_text=("CONTROL", "TREATED"),
                    ),
                    latency_ms=1.0, input_tokens=10, output_tokens=5, total_tokens=15,
                )

        apply_domain_migrations(TEST_DATABASE_URL)
        payload = build_representative_visual_deck()
        responses = []
        for describer in (Unavailable(), Describer()):
            with TestClient(create_app(
                    repository=ConversationRepository(TEST_DATABASE_URL),
                document_repository=DocumentRepository(TEST_DATABASE_URL),
                embedder=_FakeEmbedder(), visual_describer=describer,
            )) as client:
                session_id = client.post("/api/sessions").json()["session_id"]
                responses.append(client.post(
                    f"/api/documents?filename=figure.pptx&session_id={session_id}",
                    content=payload,
                    headers={"Content-Type": "application/octet-stream"},
                ))
        text_only, described = (item.json() for item in responses)
        self.assertEqual([item.status_code for item in responses], [200, 200])
        self.assertEqual(text_only["document_version"], described["document_version"])
        self.assertNotEqual(text_only["index_version"], described["index_version"])
        self.assertEqual(text_only["visuals"]["described"], 0)
        self.assertEqual(described["visuals"]["described"], 1)

        chunk_set, vectors, model_id = DocumentRepository(TEST_DATABASE_URL).load_chunk_set(
            document_version=described["document_version"],
            tenant_id="development-tenant", user_id="development-user",
        )
        derived = [item for item in chunk_set.chunks if item.evidence_kind == "derived"]
        self.assertEqual(chunk_set.index_version, described["index_version"])
        self.assertEqual(len(chunk_set.chunks), described["chunk_count"])
        self.assertEqual(len(derived), 1)
        self.assertFalse(derived[0].supports_quantitative_claims)
        self.assertEqual(len(vectors), len(chunk_set.chunks))
        self.assertEqual(model_id, "fixture-embedding")
        with psycopg.connect(TEST_DATABASE_URL) as connection:
            record = connection.execute(
                """SELECT model_id, prompt_version, description->>'literal_description'
                   FROM derived_visual_evidence
                   WHERE document_version_id = %s AND evidence_id = %s""",
                (described["document_version"], derived[0].evidence[0].evidence_id),
            ).fetchone()
            stored_versions = {row[0] for row in connection.execute(
                "SELECT DISTINCT index_version FROM chunks WHERE document_version_id = %s",
                (described["document_version"],),
            ).fetchall()}
        self.assertEqual(
            record, ("fixture-vlm", "fixture-prompt-v1", "Control and treated panels side by side.")
        )
        self.assertLessEqual(
            {text_only["index_version"], described["index_version"]}, stored_versions
        )

    def test_raw_events_are_ordered_and_tenant_scoped_across_connections(self) -> None:
        assert TEST_DATABASE_URL is not None
        apply_domain_migrations(TEST_DATABASE_URL)
        repository = ConversationRepository(TEST_DATABASE_URL)
        session_id = f"events-{uuid4()}"
        owner = {"tenant_id": "tenant-a", "user_id": "user-a"}
        repository.create_session(session_id=session_id, **owner)

        first = repository.append_event(
            session_id=session_id,
            role="user",
            event_type="message",
            content="Explain slide 4.",
            payload={"target_id": "slide-4"},
            **owner,
        )
        second = repository.append_event(
            session_id=session_id,
            role="assistant",
            event_type="explanation",
            content="The slide describes the attention mechanism.",
            payload={"evidence_ids": ["slide-4:text-2"]},
            **owner,
        )

        reopened = ConversationRepository(TEST_DATABASE_URL)
        events = reopened.list_events(session_id=session_id, **owner)
        self.assertEqual([event.sequence_no for event in events], [0, 1])
        self.assertEqual(events[0].event_id, first.event_id)
        self.assertEqual(events[1].event_id, second.event_id)
        self.assertEqual(events[1].payload["evidence_ids"], ["slide-4:text-2"])

        with self.assertRaises(SessionAccessError):
            reopened.list_events(
                session_id=session_id,
                tenant_id="tenant-b",
                user_id="user-a",
            )
        with self.assertRaises(SessionAccessError):
            reopened.append_event(
                session_id=session_id,
                tenant_id="tenant-a",
                user_id="user-b",
                role="user",
                event_type="message",
                content="Attempted cross-user access",
            )
        with self.assertRaises(SessionAccessError):
            reopened.create_session(
                session_id=session_id,
                tenant_id="tenant-b",
                user_id="user-b",
            )

        authorized = reopened.require_session(session_id=session_id, **owner)
        self.assertEqual(authorized.tenant_id, "tenant-a")

class _FakeEmbedder:
    async def embed_documents(self, texts):
        return EmbeddingBatch(
            "fixture-embedding", 768,
            tuple(tuple([float(index + 1)] * 768) for index, _ in enumerate(texts)),
            1.0,
        )

    async def embed_query(self, text):
        return EmbeddingBatch("fixture-embedding", 768, (tuple([1.0] * 768),), 1.0)


class EventLoopTests(unittest.TestCase):
    def test_uvicorn_builds_a_selector_loop_from_the_documented_setting(self) -> None:
        config = Config("backend.app.api:app", loop="backend.app.event_loop:selector_loop_factory")
        loop = config.get_loop_factory()()
        try:
            self.assertIsInstance(loop, asyncio.SelectorEventLoop)
            self.assertEqual(loop.run_until_complete(asyncio.sleep(0, result="ran")), "ran")
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
