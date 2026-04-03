CREATE TABLE IF NOT EXISTS rl_predictions (
    id BIGSERIAL PRIMARY KEY,
    model_type VARCHAR(30) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20),
    prediction JSONB NOT NULL,
    actual_outcome JSONB,
    counterfactual_pnl DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_predictions_ts
    ON rl_predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rl_predictions_model
    ON rl_predictions (model_type, timestamp DESC);
