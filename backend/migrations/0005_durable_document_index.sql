CREATE TABLE IF NOT EXISTS source_blocks (
    block_id text NOT NULL,
    document_version_id text NOT NULL REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    parent_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    block_type text NOT NULL,
    original_text text NOT NULL,
    location jsonb NOT NULL,
    hierarchy_level integer,
    PRIMARY KEY (document_version_id, block_id),
    UNIQUE (document_version_id, ordinal)
);

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS embedding_model_id text,
    ADD COLUMN IF NOT EXISTS embedded_at timestamptz;

CREATE INDEX IF NOT EXISTS source_blocks_document_version_idx
    ON source_blocks (document_version_id, ordinal);
