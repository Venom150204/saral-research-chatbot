"""SARAL API: deck upload, the conversational agent (streamed), sources and the static UI."""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from .ingestion import ExtractionError, ExtractionQualityError, UnsupportedFormatError
from .storage import (
    ArtifactRepository,
    ConversationRepository,
    SessionAccessError,
    TurnConflictError,
    apply_domain_migrations,
    database_url_from_env,
)
from .storage import DocumentRepository, InMemoryAssetStore
from .embeddings import (
    Embedder, FastEmbedder, GeminiEmbedding2, default_query_instruction,
)
from .visuals import enrich_document_visuals
from .visuals.providers.gemini import GeminiVisionDescriber, GoogleGenAITransport
from .generation.providers.gemini import (
    GeminiGroundedGenerator,
    GoogleGenAIGenerationTransport,
)
from .editing import GeminiArtifactEditor, artifact_outline
from .agent import AgentServices, build_agent, stream_turn
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from .chunking import location_label
from .retrieval import DeckRetriever
from .verification import GeminiSupportVerifier, GoogleVerificationTransport
from .workspace import InMemoryDocumentWorkspace
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"

MAX_DESCRIBED_VISUALS = 24
VISUAL_DESCRIPTION_CONCURRENCY = 3


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=100)
    document_version: str | None = Field(default=None, max_length=100)
    mode: Literal["lookup", "attribution"] = "lookup"
    per_channel: int = Field(default=3, ge=1, le=10)


class AgentTurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=100)
    document_id: str = Field(min_length=1, max_length=100)
    document_version: str = Field(min_length=1, max_length=100)
    message: str | None = Field(default=None, min_length=1, max_length=4_000)
    resume: str | None = Field(default=None, min_length=1, max_length=1_000)


def _chat_model(model_id: str, api_key: str | None):
    """The agent's chat model: low thinking for a fast first token; retries and the
    fallback are handled by the agent's middleware, not inside the client."""
    return ChatGoogleGenerativeAI(
        model=model_id, google_api_key=api_key, thinking_level="low",
        timeout=45, max_retries=0,
    )


def _model_setting(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


def create_app(
    *,
    workspace: InMemoryDocumentWorkspace | None = None,
    repository: ConversationRepository | None = None,
    document_repository: DocumentRepository | None = None,
    embedder: Embedder | None = None,
    generator: Any | None = None,
    verifier: Any | None = None,
    artifact_repository: ArtifactRepository | None = None,
    editor: Any | None = None,
    visual_describer: Any | None = None,
    agent: Any | None = None,
    frontend_dir: Path | None = None,
    use_environment_repository: bool = True,
) -> FastAPI:
    load_dotenv()
    text_provider = "gemini"
    text_model = os.getenv("SARAL_TEXT_MODEL", "gemini-3.5-flash-lite").strip()
    if not text_model:
        raise ValueError("SARAL_TEXT_MODEL cannot be blank")
    gemini_key = os.getenv("GEMINI_API_KEY")
    generation_transport = GoogleGenAIGenerationTransport(api_key=gemini_key)
    verification_transport = GoogleVerificationTransport(api_key=gemini_key)
    generation_model = _model_setting("SARAL_WRITER_MODEL", text_model)
    verifier_model = _model_setting("SARAL_VERIFIER_MODEL", text_model)
    configured_workspace = workspace or InMemoryDocumentWorkspace()
    configured_repository = repository
    configured_document_repository = document_repository
    database_url = database_url_from_env()
    if configured_repository is None and database_url and use_environment_repository:
        apply_domain_migrations(database_url)
        configured_repository = ConversationRepository(database_url)
    if configured_document_repository is None and database_url and use_environment_repository:
        configured_document_repository = DocumentRepository(database_url)
    if embedder is not None:
        configured_embedder = embedder
        embedding_provider = "injected"
        embedding_model = type(embedder).__name__
    else:
        embedding_provider = os.getenv("SARAL_EMBEDDING_PROVIDER", "local").strip().lower()
        if embedding_provider == "local":
            embedding_model = os.getenv(
                "SARAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
            )
            configured_embedder = FastEmbedder(
                model_id=embedding_model,
                cache_dir=os.getenv(
                    "SARAL_EMBEDDING_CACHE_DIR", ".cache/fastembed"
                ),
                query_instruction=default_query_instruction(embedding_model),
            )
        elif embedding_provider == "gemini":
            embedding_model = os.getenv(
                "SARAL_EMBEDDING_MODEL", "gemini-embedding-2"
            )
            configured_embedder = GeminiEmbedding2(
                api_key=os.getenv("GEMINI_API_KEY"), model_id=embedding_model
            )
        else:
            raise ValueError(
                "SARAL_EMBEDDING_PROVIDER must be either 'local' or 'gemini'"
            )
    configured_generator = generator or GeminiGroundedGenerator(
        generation_transport, model_id=generation_model,
    )
    configured_editor = editor or GeminiArtifactEditor(
        generation_transport, model_id=generation_model,
    )
    configured_verifier = verifier or GeminiSupportVerifier(
        verification_transport, model_id=verifier_model,
    )
    configured_artifact_repository = artifact_repository
    if configured_artifact_repository is None and database_url and use_environment_repository:
        configured_artifact_repository = ArtifactRepository(database_url)
    visual_mode = os.getenv("SARAL_VISUAL_DESCRIPTIONS", "on").strip().lower()
    if visual_mode not in {"on", "off"}:
        raise ValueError("SARAL_VISUAL_DESCRIPTIONS must be either 'on' or 'off'")
    vision_model = os.getenv("SARAL_VISION_MODEL", "").strip() or text_model
    configured_describer = visual_describer
    if configured_describer is None and visual_mode == "on" and gemini_key:
        configured_describer = GeminiVisionDescriber(
            GoogleGenAITransport(api_key=gemini_key), model_id=vision_model
        )
    agent_model = _model_setting("SARAL_AGENT_MODEL", "gemini-3.5-flash-lite")
    agent_fallback_model = (os.getenv("SARAL_AGENT_FALLBACK_MODEL") or "gemini-3.1-flash-lite").strip()
    agent_services = AgentServices(
        document_repository=configured_document_repository,
        artifact_repository=configured_artifact_repository,
        ledger=configured_repository, embedder=configured_embedder,
        generator=configured_generator, verifier=configured_verifier, editor=configured_editor,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent, app.state.agent_error = agent, None
        stack = AsyncExitStack()
        if agent is None and database_url and use_environment_repository:
            try:
                proactor = getattr(asyncio, "ProactorEventLoop", None)
                if proactor is not None and isinstance(asyncio.get_running_loop(), proactor):
                    raise RuntimeError("psycopg cannot use the ProactorEventLoop")
                pool = AsyncConnectionPool(
                    database_url, min_size=1, max_size=4, open=False,
                    check=AsyncConnectionPool.check_connection,
                    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                )
                await pool.open(wait=True, timeout=15)
                stack.push_async_callback(pool.close)
                saver = AsyncPostgresSaver(pool)
                await saver.setup()
                app.state.agent = build_agent(
                    services=agent_services,
                    model=_chat_model(agent_model, gemini_key),
                    fallback_model=(
                        _chat_model(agent_fallback_model, gemini_key)
                        if agent_fallback_model and agent_fallback_model != agent_model else None
                    ),
                    checkpointer=saver,
                )
            except Exception as error:
                await stack.aclose()
                windows_loop = "ProactorEventLoop" in str(error)
                app.state.agent_error = (
                    "Start the server with --loop backend.app.event_loop:selector_loop_factory "
                    "(Windows needs the selector event loop for the async Postgres driver)."
                    if windows_loop else f"The agent could not start: {type(error).__name__}"
                )
                logger.warning("Agent unavailable: %s", app.state.agent_error, exc_info=not windows_loop)
        try:
            yield
        finally:
            await stack.aclose()

    app = FastAPI(title="SARAL", version="0.2.0", lifespan=lifespan)
    frontend = frontend_dir or FRONTEND_DIR
    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "surface": "connected-development",
            "postgres_configured": configured_repository is not None,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "text_provider": text_provider,
            "generation_model": generation_model,
            "verifier_model": verifier_model,
            "agent_ready": getattr(app.state, "agent", None) is not None,
            "agent_model": agent_model,
            "agent_fallback_model": agent_fallback_model,
            "agent_error": getattr(app.state, "agent_error", None),
            "visual_descriptions": "on" if configured_describer is not None else "off",
            "vision_model": vision_model if configured_describer is not None else None,
        }

    @app.get("/", response_model=None)
    def index() -> FileResponse | HTMLResponse:
        page = frontend / "index.html"
        if not page.is_file():
            return HTMLResponse(
                "<!doctype html><title>SARAL</title><p>The interface is not built yet. Run "
                "<code>npm install</code> and <code>npm run build</code> in <code>web/</code>, "
                "then reload.</p>",
                headers={"Cache-Control": "no-store"},
            )
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    @app.post("/api/chat/stream")
    async def stream_chat(request: AgentTurnRequest) -> StreamingResponse:
        """One conversational turn with the agent, streamed as server-sent events.
        Send `message` for a new turn, or `resume` with the same turn_id to answer a
        question the agent asked."""
        running = getattr(app.state, "agent", None)
        if running is None:
            raise HTTPException(status_code=503, detail=(
                getattr(app.state, "agent_error", None) or "The agent is not enabled."
            ))
        if (request.message is None) == (request.resume is None):
            raise HTTPException(status_code=422, detail="Send either a message or a resume answer.")

        async def events():
            try:
                async for event in stream_turn(
                    agent=running, services=agent_services, agent_model_id=agent_model,
                    session_id=request.session_id, turn_id=request.turn_id,
                    tenant_id="development-tenant", user_id="development-user",
                    document_id=request.document_id, document_version=request.document_version,
                    message=request.message, resume=request.resume,
                ):
                    yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except (SessionAccessError, TurnConflictError) as error:
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.post("/api/sessions")
    def create_session() -> dict[str, str]:
        repo = _require_repository(configured_repository)
        session_id = f"session-{uuid4()}"
        session = repo.create_session(
            session_id=session_id,
            tenant_id="development-tenant",
            user_id="development-user",
        )
        return {"session_id": session.session_id}

    @app.get("/api/sessions/{session_id}")
    def resume_session(session_id: str) -> dict[str, Any]:
        repo = _require_repository(configured_repository)
        try:
            session = repo.require_session(
                session_id=session_id,
                tenant_id="development-tenant",
                user_id="development-user",
            )
            events = repo.list_events(
                session_id=session_id,
                tenant_id="development-tenant",
                user_id="development-user",
            )
        except SessionAccessError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "session_id": session.session_id,
            "active_document_version_id": session.active_document_version_id,
            "events": [asdict(event) for event in events],
            "active_document": _active_document_summary(
                configured_document_repository,
                session.active_document_version_id,
            ),
            "active_artifact": (
                configured_artifact_repository.load_active(
                    session_id=session_id,
                    tenant_id="development-tenant",
                    user_id="development-user",
                )
                if configured_artifact_repository is not None else None
            ),
        }

    @app.post("/api/documents")
    async def upload_document(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        session_id: str | None = Query(default=None, max_length=100),
    ) -> dict[str, Any]:
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="uploaded file is empty")
        asset_store = InMemoryAssetStore() if configured_describer is not None else None
        try:
            document = configured_workspace.extract(
                payload, filename=filename, asset_store=asset_store
            )
        except (
            ExtractionError,
            ExtractionQualityError,
            UnsupportedFormatError,
            ValueError,
        ) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        pictures = [item for item in document.visuals if item.visual_type == "picture"]
        visual_evidence: tuple = ()
        visual_summary: dict[str, Any] = {
            "descriptions": "on" if configured_describer is not None else "off",
            "pictures_found": len(pictures),
            "described": 0,
            "skipped": [],
            "failed": [],
        }
        if configured_describer is not None and pictures:
            try:
                enrichment = await enrich_document_visuals(
                    document,
                    asset_store=asset_store,
                    describer=configured_describer,
                    continue_on_error=True,
                    skip_decorative=True,
                    max_visuals=MAX_DESCRIBED_VISUALS,
                    max_concurrency=VISUAL_DESCRIPTION_CONCURRENCY,
                )
            except Exception as error:
                logger.exception("Visual description failed for %s", document.document_id)
                visual_summary["failed"] = [
                    {"slide_number": item.slide_number, "reason": type(error).__name__}
                    for item in pictures
                ]
            else:
                visual_evidence = enrichment.evidence
                visual_summary.update(
                    described=len(enrichment.evidence),
                    skipped=[
                        {"slide_number": item.slide_number, "reason": item.reason}
                        for item in enrichment.issues if item.status == "skipped"
                    ],
                    failed=[
                        {"slide_number": item.slide_number, "reason": item.reason}
                        for item in enrichment.issues if item.status == "failed"
                    ],
                    model_id=next(
                        (item.model_id for item in enrichment.evidence), None
                    ),
                    input_tokens=sum(
                        item.input_tokens or 0 for item in enrichment.telemetry
                    ),
                    output_tokens=sum(
                        item.output_tokens or 0 for item in enrichment.telemetry
                    ),
                )
        try:
            stored = configured_workspace.store(
                document, visual_evidence=visual_evidence
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        persistence = "process_memory"
        if configured_document_repository is not None:
            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="session_id is required for durable document indexing",
                )
            try:
                batch = await configured_embedder.embed_documents(
                    tuple(chunk.search_text for chunk in stored.chunk_set.chunks)
                )
                configured_document_repository.save_index(
                    document=document,
                    chunks=stored.chunk_set,
                    vectors=batch.vectors,
                    embedding_model_id=batch.model_id,
                    tenant_id="development-tenant",
                    user_id="development-user",
                    session_id=session_id,
                    visual_evidence=visual_evidence,
                )
            except SessionAccessError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except Exception as error:
                logger.exception("Durable indexing failed for %s", document.document_id)
                raise HTTPException(
                    status_code=503,
                    detail="Document embedding or durable indexing failed.",
                ) from error
            persistence = "postgresql_pgvector"
        described_ids = {item.visual_id for item in visual_evidence}
        uninterpreted_slides = {
            item.slide_number for item in document.visuals
            if item.visual_id not in described_ids
        }
        return {
            "document_id": document.document_id,
            "document_version": document.document_version,
            "index_version": stored.chunk_set.index_version,
            "filename": document.filename,
            "source_kind": document.source_kind,
            "unit_count": document.unit_count,
            "block_count": len(document.blocks),
            "visual_count": len(document.visuals),
            "visuals": visual_summary,
            "chunk_count": len(stored.chunk_set.chunks),
            "warnings": [
                asdict(warning) for warning in document.warnings
                if warning.code != "visual_meaning_not_interpreted"
                or warning.slide_number in uninterpreted_slides
            ],
            "persistence": persistence,
        }

    @app.post("/api/documents/{document_id}/retrieve")
    async def retrieve_evidence(
        document_id: str, request: RetrieveRequest
    ) -> dict[str, Any]:
        try:
            if configured_document_repository is not None and request.document_version:
                chunk_set, vectors, stored_model_id = configured_document_repository.load_chunk_set(
                    document_version=request.document_version,
                    tenant_id="development-tenant",
                    user_id="development-user",
                )
                if chunk_set.document_id != document_id:
                    raise KeyError("document version does not match document")
                retriever = DeckRetriever(
                    chunk_set, vectors, embedder=configured_embedder,
                    embedding_model_id=stored_model_id,
                )
                document_version = request.document_version
                if request.mode == "attribution":
                    retrieval_mode = "pooled_candidates"
                    ranked = [
                        (
                            position,
                            sum(
                                1 / (60 + rank)
                                for rank in (item.keyword_rank, item.embedding_rank)
                                if rank is not None
                            ),
                            item.keyword_rank, item.embedding_rank, item.chunk,
                        )
                        for position, item in enumerate(
                            await retriever.candidates(
                                request.query, per_channel=request.per_channel
                            ),
                            start=1,
                        )
                    ]
                else:
                    retrieval_mode = "hybrid_rrf"
                    ranked = [
                        (item.rank, item.rrf_score, item.lexical_rank, item.dense_rank, item.hit.chunk)
                        for item in await retriever.lookup(
                            request.query, top_k=request.top_k
                        )
                    ]
            else:
                stored = configured_workspace.get(document_id)
                document_version = stored.document.document_version
                retrieval_mode = "lexical_diagnostic"
                ranked = [
                    (item.rank, item.score, item.rank, None, item.chunk)
                    for item in configured_workspace.retrieve(
                        document_id, query=request.query, top_k=request.top_k
                    )
                ]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "document_id": document_id,
            "document_version": document_version,
            "query": request.query,
            "retrieval_mode": retrieval_mode,
            "hits": [
                {
                    "rank": rank,
                    "score": score,
                    "lexical_rank": keyword_rank,
                    "dense_rank": embedding_rank,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text,
                    "evidence": [asdict(evidence) for evidence in chunk.evidence],
                }
                for rank, score, keyword_rank, embedding_rank, chunk in ranked
            ],
        }

    @app.get("/api/documents/{document_id}/sources")
    def document_sources(
        document_id: str,
        document_version: str | None = Query(default=None, max_length=100),
    ) -> dict[str, Any]:
        """Every citable unit of a document, so a citation can show its source line."""
        try:
            if configured_document_repository is not None and document_version:
                chunk_set, _, _ = configured_document_repository.load_chunk_set(
                    document_version=document_version,
                    tenant_id="development-tenant",
                    user_id="development-user",
                )
                if chunk_set.document_id != document_id:
                    raise KeyError("document version does not match document")
            else:
                chunk_set = configured_workspace.get(document_id).chunk_set
        except (KeyError, LookupError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "document_id": chunk_set.document_id,
            "document_version": chunk_set.document_version,
            "sources": [
                {
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.chunk_type,
                    "slide_number": next(
                        (ref.slide_number for ref in chunk.evidence
                         if ref.slide_number is not None),
                        None,
                    ),
                    "page_number": next(
                        (ref.page_number for ref in chunk.evidence
                         if ref.page_number is not None),
                        None,
                    ),
                    "location": location_label(chunk.evidence),
                    "section": chunk.parent_id,
                    "text": chunk.text,
                    "image_description": chunk.evidence_kind == "derived",
                }
                for chunk in chunk_set.chunks
            ],
        }

    return app


app = create_app()


def _require_repository(
    repository: ConversationRepository | None,
) -> ConversationRepository:
    if repository is None:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL persistence is not configured; set SARAL_DATABASE_URL",
        )
    return repository


def _active_document_summary(
    repository: DocumentRepository | None, document_version: str | None
) -> dict[str, Any] | None:
    if repository is None or document_version is None:
        return None
    chunk_set, _, model_id = repository.load_chunk_set(
        document_version=document_version,
        tenant_id="development-tenant",
        user_id="development-user",
    )
    slides = {ref.slide_number for chunk in chunk_set.chunks for ref in chunk.evidence if ref.slide_number}
    pages = {ref.page_number for chunk in chunk_set.chunks for ref in chunk.evidence if ref.page_number}
    kind, units = ("slide_deck", len(slides)) if slides else ("document", len(pages)) if pages else ("summary", None)
    return {
        "document_id": chunk_set.document_id,
        "document_version": chunk_set.document_version,
        "embedding_model_id": model_id,
        "filename": repository.display_name(
            document_id=chunk_set.document_id, tenant_id="development-tenant", user_id="development-user",
        ),
        "source_kind": kind,
        "unit_count": units,
    }
