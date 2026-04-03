BEGIN;

CREATE TABLE IF NOT EXISTS crypto_metrics (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    funding_rate DOUBLE PRECISION,
    open_interest DOUBLE PRECISION CHECK (open_interest >= 0.0),
    btc_dominance DOUBLE PRECISION CHECK (btc_dominance BETWEEN 0.0 AND 100.0),
    fear_greed_index INTEGER CHECK (fear_greed_index BETWEEN 0 AND 100),
    tvl DOUBLE PRECISION CHECK (tvl >= 0.0),
    tvl_change_24h DOUBLE PRECISION,
    exchange_reserves DOUBLE PRECISION CHECK (exchange_reserves >= 0.0),
    liquidations_24h DOUBLE PRECISION CHECK (liquidations_24h >= 0.0),
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (
        funding_rate IS NOT NULL OR
        open_interest IS NOT NULL OR
        fear_greed_index IS NOT NULL OR
        tvl IS NOT NULL
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_crypto_metrics_unique
    ON crypto_metrics (symbol, timestamp, source);

CREATE INDEX IF NOT EXISTS idx_crypto_metrics_ts
    ON crypto_metrics (symbol, timestamp DESC);

COMMENT ON COLUMN crypto_metrics.funding_rate IS 'Perpetual futures funding rate, positive = longs pay shorts';
COMMENT ON COLUMN crypto_metrics.btc_dominance IS 'BTC market cap dominance percentage [0-100]';
COMMENT ON COLUMN crypto_metrics.fear_greed_index IS 'Crypto Fear & Greed Index [0-100], 0 = extreme fear';
COMMENT ON COLUMN crypto_metrics.tvl IS 'Total Value Locked in DeFi protocols, USD';
COMMENT ON COLUMN crypto_metrics.liquidations_24h IS 'Total liquidation volume in 24h, USD';

COMMIT;
