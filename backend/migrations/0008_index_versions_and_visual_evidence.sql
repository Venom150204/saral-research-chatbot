-- A document version can be indexed more than once: the extraction, the chunking
-- configuration, or the set of image descriptions may change. Each such result is
-- an immutable index version. Older index versions are kept because committed
-- artifacts cite their chunks.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS index_version text NOT NULL DEFAULT 'legacy';

ALTER TABLE chunks
    DROP CONSTRAINT IF EXISTS chunks_document_version_id_ordinal_key;

ALTER TABLE chunks
    ADD CONSTRAINT chunks_document_index_ordinal_key
    UNIQUE (document_version_id, index_version, ordinal);

ALTER TABLE source_blocks
    ADD COLUMN IF NOT EXISTS index_version text NOT NULL DEFAULT 'legacy';

ALTER TABLE source_blocks
    DROP CONSTRAINT IF EXISTS source_blocks_document_version_id_ordinal_key;

ALTER TABLE source_blocks
    DROP CONSTRAINT IF EXISTS source_blocks_pkey;

ALTER TABLE source_blocks
    ADD CONSTRAINT source_blocks_pkey
    PRIMARY KEY (document_version_id, index_version, block_id);

ALTER TABLE source_blocks
    ADD CONSTRAINT source_blocks_document_index_ordinal_key
    UNIQUE (document_version_id, index_version, ordinal);

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS active_index_version text;

UPDATE document_versions v
SET active_index_version = 'legacy'
WHERE v.active_index_version IS NULL
  AND EXISTS (
      SELECT 1 FROM chunks c WHERE c.document_version_id = v.document_version_id
  );

-- One row per model-written image description, with the provenance needed to
-- reproduce or audit it. The description text itself is also indexed as a chunk.
CREATE TABLE IF NOT EXISTS derived_visual_evidence (
    evidence_id text NOT NULL,
    document_version_id text NOT NULL
        REFERENCES document_versions(document_version_id) ON DELETE CASCADE,
    visual_id text NOT NULL,
    asset_sha256 text NOT NULL CHECK (length(asset_sha256) = 64),
    slide_number integer NOT NULL CHECK (slide_number > 0),
    shape_id integer NOT NULL,
    model_id text NOT NULL,
    prompt_version text NOT NULL,
    description jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_version_id, evidence_id)
);
