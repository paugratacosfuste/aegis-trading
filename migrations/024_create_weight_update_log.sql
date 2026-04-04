-- Weight update audit log for daily Bayesian updates
CREATE TABLE IF NOT EXISTS weight_update_log (
    id BIGSERIAL PRIMARY KEY,
    agent_id VARCHAR(40) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,
    old_weight DOUBLE PRECISION NOT NULL,
    new_weight DOUBLE PRECISION NOT NULL,
    hit_rate DOUBLE PRECISION,
    information_coefficient DOUBLE PRECISION,
    composite_score DOUBLE PRECISION,
    n_signals INTEGER NOT NULL,
    update_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weight_log_agent
    ON weight_update_log (agent_id, update_date DESC);

CREATE INDEX IF NOT EXISTS idx_weight_log_date
    ON weight_update_log (update_date DESC);
