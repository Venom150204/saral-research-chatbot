-- A running summary of the turns that no longer fit the recent-conversation budget,
-- with the lasting preferences the user stated (audience, length, style, focus).
-- One row per conversation; it only moves forward as more turns are folded in.
CREATE TABLE conversation_memory (
    session_id text PRIMARY KEY
        REFERENCES conversation_sessions (session_id) ON DELETE CASCADE,
    covers_through_sequence bigint NOT NULL CHECK (covers_through_sequence >= 0),
    summary text NOT NULL,
    preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
    model_id text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
