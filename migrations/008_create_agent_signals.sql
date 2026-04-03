CREATE TABLE IF NOT EXISTS agent_signals (
    id BIGSERIAL PRIMARY KEY,
    agent_id VARCHAR(40) NOT NULL,
    agent_type VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    direction DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    timeframe VARCHAR(5),
    expected_holding_period VARCHAR(10),
    entry_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    reasoning JSONB,
    features_used JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_signals_agent_ts
    ON agent_signals (agent_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_signals_symbol_ts
    ON agent_signals (symbol, timestamp DESC);
