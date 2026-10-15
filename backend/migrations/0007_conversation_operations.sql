CREATE TABLE conversation_operations (
    session_id text NOT NULL,
    turn_id text NOT NULL,
    step_id text NOT NULL,
    capability text NOT NULL,
    status text NOT NULL CHECK (status IN ('running', 'completed')),
    attempt_count integer NOT NULL DEFAULT 1 CHECK (attempt_count > 0),
    result jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id, step_id),
    FOREIGN KEY (session_id, turn_id)
        REFERENCES conversation_turns (session_id, turn_id)
        ON DELETE CASCADE,
    CHECK (
        (status = 'running' AND result IS NULL AND completed_at IS NULL)
        OR
        (status = 'completed' AND result IS NOT NULL AND completed_at IS NOT NULL)
    )
);

CREATE INDEX conversation_operations_turn_status_idx
    ON conversation_operations (session_id, turn_id, status);
