CREATE TABLE IF NOT EXISTS fundamentals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    pe_trailing DOUBLE PRECISION,
    pe_forward DOUBLE PRECISION,
    pb DOUBLE PRECISION,
    ps DOUBLE PRECISION,
    ev_ebitda DOUBLE PRECISION,
    peg DOUBLE PRECISION,
    revenue_growth_yoy DOUBLE PRECISION,
    eps_growth DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    roa DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    free_cash_flow_margin DOUBLE PRECISION,
    dividend_yield DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    sector VARCHAR(50),
    source VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fundamentals_unique
    ON fundamentals (symbol, date, source);
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol
    ON fundamentals (symbol, date DESC);
