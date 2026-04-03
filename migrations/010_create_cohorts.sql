CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id VARCHAR(40) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cohorts_status
    ON cohorts (status);
