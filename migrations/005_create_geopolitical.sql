CREATE TABLE IF NOT EXISTS geopolitical (
    id BIGSERIAL PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(30) NOT NULL,
    category VARCHAR(20) NOT NULL,
    severity DOUBLE PRECISION NOT NULL,
    affected_sectors TEXT[],
    affected_regions TEXT[],
    raw_text TEXT,
    sentiment_score DOUBLE PRECISION,
    half_life_hours INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_geopolitical_event
    ON geopolitical (event_id);
CREATE INDEX IF NOT EXISTS idx_geopolitical_ts
    ON geopolitical (timestamp DESC);
