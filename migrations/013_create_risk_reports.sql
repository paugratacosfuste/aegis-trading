CREATE TABLE IF NOT EXISTS risk_reports (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_value DOUBLE PRECISION NOT NULL,
    cash_pct DOUBLE PRECISION,
    invested_pct DOUBLE PRECISION,
    open_positions INTEGER,
    long_positions INTEGER,
    short_positions INTEGER,
    daily_pnl DOUBLE PRECISION,
    weekly_pnl DOUBLE PRECISION,
    sharpe_30d DOUBLE PRECISION,
    win_rate_20trades DOUBLE PRECISION,
    active_breakers TEXT[],
    warning_breakers TEXT[],
    data_staleness_flags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_reports_ts
    ON risk_reports (timestamp DESC);
