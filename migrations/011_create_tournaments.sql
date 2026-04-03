CREATE TABLE IF NOT EXISTS tournaments (
    id BIGSERIAL PRIMARY KEY,
    week_start DATE NOT NULL,
    cohort_id VARCHAR(40) NOT NULL,
    sharpe DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    total_trades INTEGER DEFAULT 0,
    net_pnl DOUBLE PRECISION,
    rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tournaments_week
    ON tournaments (week_start DESC);
CREATE INDEX IF NOT EXISTS idx_tournaments_cohort
    ON tournaments (cohort_id, week_start DESC);
