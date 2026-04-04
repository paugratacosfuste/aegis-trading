BEGIN;

ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS generation INTEGER DEFAULT 0;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS parent_cohort_id VARCHAR(40);
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS relegation_count INTEGER DEFAULT 0;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS burn_in_start TIMESTAMPTZ;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS evaluation_start TIMESTAMPTZ;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS relegated_at TIMESTAMPTZ;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS retired_at TIMESTAMPTZ;
ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS virtual_capital NUMERIC(18, 6) DEFAULT 100000.0;

COMMENT ON COLUMN cohorts.generation IS 'Mutation generation (0 = original template)';
COMMENT ON COLUMN cohorts.parent_cohort_id IS 'Cohort this was mutated from (NULL for originals)';
COMMENT ON COLUMN cohorts.relegation_count IS 'Number of times relegated (retired at 2)';
COMMENT ON COLUMN cohorts.virtual_capital IS 'Current virtual capital balance in USD';

COMMIT;
