CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    owner_user_id text NOT NULL,
    source_kind text NOT NULL CHECK (source_kind IN ('pptx', 'summary')),
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_versions (
    document_version_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    content_sha256 text NOT NULL CHECK (length(content_sha256) = 64),
    parser_version text NOT NULL,
    source_uri text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, content_sha256)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    document_version_id text NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    parent_chunk_id text REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    original_text text NOT NULL,
    search_text text NOT NULL,
    embedding vector(768),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_version_id, ordinal)
);

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    user_id text NOT NULL,
    active_document_version_id text REFERENCES document_versions(document_version_id) ON DELETE SET NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_events (
    event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    sequence_no bigint NOT NULL CHECK (sequence_no >= 0),
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    event_type text NOT NULL,
    content text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, sequence_no)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    document_version_id text NOT NULL REFERENCES document_versions(document_version_id) ON DELETE RESTRICT,
    artifact_kind text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifact_revisions (
    revision_id text PRIMARY KEY,
    artifact_id text NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    version integer NOT NULL CHECK (version > 0),
    parent_revision_id text REFERENCES artifact_revisions(revision_id) ON DELETE RESTRICT,
    content jsonb NOT NULL,
    edit_instruction text,
    change_reason text,
    model_id text NOT NULL,
    prompt_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (artifact_id, version)
);

CREATE TABLE IF NOT EXISTS citations (
    citation_id text PRIMARY KEY,
    revision_id text NOT NULL REFERENCES artifact_revisions(revision_id) ON DELETE CASCADE,
    unit_id text NOT NULL,
    chunk_id text NOT NULL REFERENCES chunks(chunk_id) ON DELETE RESTRICT,
    evidence_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (revision_id, unit_id, chunk_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS documents_tenant_idx
    ON documents (tenant_id, owner_user_id);
CREATE INDEX IF NOT EXISTS chunks_document_version_idx
    ON chunks (document_version_id, ordinal);
CREATE INDEX IF NOT EXISTS conversation_sessions_tenant_idx
    ON conversation_sessions (tenant_id, user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS conversation_events_session_idx
    ON conversation_events (session_id, sequence_no);
CREATE INDEX IF NOT EXISTS artifacts_session_idx
    ON artifacts (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS citations_revision_unit_idx
    ON citations (revision_id, unit_id);

