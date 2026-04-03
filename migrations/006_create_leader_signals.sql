CREATE TABLE IF NOT EXISTS leader_signals (
    id BIGSERIAL PRIMARY KEY,
    leader VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    statement_type VARCHAR(20) NOT NULL,
    policy_dimensions JSONB,
    half_life_hours INTEGER,
    affected_symbols TEXT[],
    raw_text TEXT,
    sentiment_score DOUBLE PRECISION,
    source VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_leader_signals_ts
    ON leader_signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_leader_signals_leader
    ON leader_signals (leader, timestamp DESC);
