ALTER TABLE conversation_turns
    ADD COLUMN IF NOT EXISTS execution_plan jsonb,
    ADD COLUMN IF NOT EXISTS planner_model_id text,
    ADD COLUMN IF NOT EXISTS planner_prompt_version text,
    ADD COLUMN IF NOT EXISTS planned_at timestamptz;

ALTER TABLE conversation_turns
    ADD CONSTRAINT conversation_turns_plan_metadata_consistent CHECK (
        (execution_plan IS NULL AND planner_model_id IS NULL
            AND planner_prompt_version IS NULL AND planned_at IS NULL)
        OR
        (execution_plan IS NOT NULL AND planner_model_id IS NOT NULL
            AND planner_prompt_version IS NOT NULL AND planned_at IS NOT NULL)
    );
