CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_ts
    ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_tf_ts
    ON market_data (symbol, timeframe, timestamp DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_unique
    ON market_data (symbol, timeframe, timestamp, source);
