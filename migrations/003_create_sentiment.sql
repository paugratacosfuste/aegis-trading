CREATE TABLE IF NOT EXISTS sentiment (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(20) NOT NULL,
    sentiment_score DOUBLE PRECISION NOT NULL,
    mention_count INTEGER DEFAULT 0,
    sentiment_velocity DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_ts
    ON sentiment (symbol, timestamp DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sentiment_unique
    ON sentiment (symbol, timestamp, source);
