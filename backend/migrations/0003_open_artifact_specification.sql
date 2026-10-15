ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS output_specification jsonb;

UPDATE artifacts
SET output_specification = jsonb_build_object('format', artifact_kind)
WHERE output_specification IS NULL;

ALTER TABLE artifacts
    ALTER COLUMN output_specification SET NOT NULL;

ALTER TABLE artifacts
    DROP COLUMN IF EXISTS artifact_kind;
