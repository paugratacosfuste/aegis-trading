BEGIN;

ALTER TABLE tournaments ADD COLUMN IF NOT EXISTS composite_score DOUBLE PRECISION;
ALTER TABLE tournaments ADD COLUMN IF NOT EXISTS profit_factor DOUBLE PRECISION;

COMMENT ON COLUMN tournaments.composite_score IS 'Weighted composite: 0.40*sharpe + 0.25*(1-max_dd) + 0.20*profit_factor + 0.15*win_rate';
COMMENT ON COLUMN tournaments.profit_factor IS 'Gross profit / gross loss for the tournament period';

COMMIT;
