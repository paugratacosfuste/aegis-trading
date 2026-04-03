CREATE TABLE IF NOT EXISTS macro (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    yield_10y DOUBLE PRECISION,
    yield_2y DOUBLE PRECISION,
    yield_spread DOUBLE PRECISION,
    vix DOUBLE PRECISION,
    vix_regime VARCHAR(10),
    dxy DOUBLE PRECISION,
    fed_rate DOUBLE PRECISION,
    cpi_latest DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_unique
    ON macro (timestamp);
CREATE INDEX IF NOT EXISTS idx_macro_ts
    ON macro (timestamp DESC);
