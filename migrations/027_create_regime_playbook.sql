-- Per-regime learned optimal settings
CREATE TABLE IF NOT EXISTS regime_playbook (
    id BIGSERIAL PRIMARY KEY,
    regime VARCHAR(30) NOT NULL,
    last_updated DATE NOT NULL,
    total_observations INTEGER NOT NULL,
    best_agent_weights JSONB NOT NULL,
    best_cohort_ids JSONB NOT NULL,
    avg_sharpe DOUBLE PRECISION,
    avg_win_rate DOUBLE PRECISION,
    worst_agent_types JSONB,
    recommended_position_size_mult DOUBLE PRECISION DEFAULT 1.0,
    recommended_max_positions INTEGER DEFAULT 10,
    recommended_confidence_threshold DOUBLE PRECISION DEFAULT 0.45,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_regime_playbook_regime
    ON regime_playbook (regime, last_updated);
