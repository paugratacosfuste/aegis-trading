CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR(64) PRIMARY KEY,
    account_type VARCHAR(10) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    direction VARCHAR(5) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    quantity DOUBLE PRECISION NOT NULL,
    position_value DOUBLE PRECISION NOT NULL,
    commission_entry DOUBLE PRECISION DEFAULT 0,
    commission_exit DOUBLE PRECISION DEFAULT 0,
    estimated_slippage DOUBLE PRECISION DEFAULT 0,
    total_costs DOUBLE PRECISION DEFAULT 0,
    gross_pnl DOUBLE PRECISION,
    net_pnl DOUBLE PRECISION,
    return_pct DOUBLE PRECISION,
    r_multiple DOUBLE PRECISION,
    holding_period_hours DOUBLE PRECISION,
    ensemble_confidence DOUBLE PRECISION NOT NULL,
    ensemble_direction DOUBLE PRECISION NOT NULL,
    agent_signals JSONB,
    regime_at_entry VARCHAR(30),
    initial_stop_loss DOUBLE PRECISION NOT NULL,
    risk_amount DOUBLE PRECISION NOT NULL,
    risk_pct_of_portfolio DOUBLE PRECISION NOT NULL,
    exit_reason VARCHAR(30),
    feature_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
    ON trades (symbol, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_account
    ON trades (account_type, entry_time DESC);
