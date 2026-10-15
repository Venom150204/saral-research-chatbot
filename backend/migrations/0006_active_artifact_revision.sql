ALTER TABLE conversation_sessions
    ADD COLUMN IF NOT EXISTS active_artifact_id text,
    ADD COLUMN IF NOT EXISTS active_artifact_version integer;

ALTER TABLE conversation_sessions
    ADD CONSTRAINT conversation_sessions_active_artifact_consistent CHECK (
        (active_artifact_id IS NULL AND active_artifact_version IS NULL)
        OR
        (active_artifact_id IS NOT NULL AND active_artifact_version IS NOT NULL
            AND active_artifact_version > 0)
    );

ALTER TABLE conversation_sessions
    ADD CONSTRAINT conversation_sessions_active_artifact_revision_fk
    FOREIGN KEY (active_artifact_id, active_artifact_version)
    REFERENCES artifact_revisions (artifact_id, version)
    DEFERRABLE INITIALLY DEFERRED;
