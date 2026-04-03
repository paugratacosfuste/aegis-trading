CREATE TABLE IF NOT EXISTS agent_weights (
    id BIGSERIAL PRIMARY KEY,
    agent_id VARCHAR(40) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    hit_rate DOUBLE PRECISION,
    information_coefficient DOUBLE PRECISION,
    rolling_sharpe DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_weights_agent
    ON agent_weights (agent_id, updated_at DESC);
