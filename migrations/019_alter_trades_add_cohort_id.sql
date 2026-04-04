BEGIN;

ALTER TABLE trades ADD COLUMN IF NOT EXISTS cohort_id VARCHAR(40) DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_trades_cohort_ts
    ON trades (cohort_id, entry_time DESC)
    WHERE cohort_id IS NOT NULL;

COMMENT ON COLUMN trades.cohort_id IS 'Lab cohort that generated this trade. NULL for production trades.';

COMMIT;
