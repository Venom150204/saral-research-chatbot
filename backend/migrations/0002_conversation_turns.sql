CREATE TABLE IF NOT EXISTS conversation_turns (
    session_id text NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    turn_id text NOT NULL,
    request_sha256 text NOT NULL CHECK (length(request_sha256) = 64),
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    user_event_id bigint REFERENCES conversation_events(event_id) ON DELETE RESTRICT,
    outcome_event_id bigint REFERENCES conversation_events(event_id) ON DELETE RESTRICT,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id)
);

CREATE INDEX IF NOT EXISTS conversation_turns_status_idx
    ON conversation_turns (status, updated_at);

