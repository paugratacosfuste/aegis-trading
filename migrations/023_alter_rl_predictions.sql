-- Add columns for shadow mode tracking
ALTER TABLE rl_predictions
    ADD COLUMN IF NOT EXISTS context_features JSONB,
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(50),
    ADD COLUMN IF NOT EXISTS mode VARCHAR(20) DEFAULT 'shadow';

CREATE INDEX IF NOT EXISTS idx_rl_predictions_mode
    ON rl_predictions (mode, model_type, timestamp DESC);
