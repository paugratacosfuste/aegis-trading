CREATE TABLE IF NOT EXISTS agent_performance (
    id BIGSERIAL PRIMARY KEY,
    agent_id VARCHAR(40) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,
    predicted_direction DOUBLE PRECISION NOT NULL,
    actual_return DOUBLE PRECISION NOT NULL,
    is_correct BOOLEAN NOT NULL,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    outcome_timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_perf_agent
    ON agent_performance (agent_id, outcome_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_perf_symbol
    ON agent_performance (symbol, outcome_timestamp DESC);
