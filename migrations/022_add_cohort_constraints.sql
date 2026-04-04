BEGIN;

-- Unique constraint: one tournament entry per cohort per week
DO $$ BEGIN
    ALTER TABLE tournaments ADD CONSTRAINT uq_tournaments_week_cohort
        UNIQUE (week_start, cohort_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Generation must be non-negative
DO $$ BEGIN
    ALTER TABLE cohorts ADD CONSTRAINT chk_generation
        CHECK (generation >= 0);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Relegation count bounded 0-2
DO $$ BEGIN
    ALTER TABLE cohorts ADD CONSTRAINT chk_relegation_count
        CHECK (relegation_count >= 0 AND relegation_count <= 2);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tournament rank must be positive
DO $$ BEGIN
    ALTER TABLE tournaments ADD CONSTRAINT chk_rank_positive
        CHECK (rank >= 1);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Profit factor guard (cap Inf at 1000)
DO $$ BEGIN
    ALTER TABLE tournaments ADD CONSTRAINT chk_profit_factor
        CHECK (profit_factor IS NULL OR (profit_factor >= 0 AND profit_factor < 1000));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Partial index for lineage queries
CREATE INDEX IF NOT EXISTS idx_cohorts_parent
    ON cohorts (parent_cohort_id)
    WHERE parent_cohort_id IS NOT NULL;

COMMENT ON CONSTRAINT uq_tournaments_week_cohort ON tournaments IS 'Prevent duplicate tournament entries per cohort per week';
COMMENT ON CONSTRAINT chk_relegation_count ON cohorts IS 'Retired at 2 relegations, cannot exceed';

COMMIT;
