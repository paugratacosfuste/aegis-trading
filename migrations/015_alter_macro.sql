BEGIN;

ALTER TABLE macro ADD COLUMN IF NOT EXISTS real_rate_proxy DOUBLE PRECISION;
ALTER TABLE macro ADD COLUMN IF NOT EXISTS dxy_trend VARCHAR(15)
    CHECK (dxy_trend IN ('strengthening', 'weakening', 'neutral'));
ALTER TABLE macro ADD COLUMN IF NOT EXISTS unemployment_rate DOUBLE PRECISION
    CHECK (unemployment_rate >= 0.0);
ALTER TABLE macro ADD COLUMN IF NOT EXISTS gdp_latest DOUBLE PRECISION;
ALTER TABLE macro ADD COLUMN IF NOT EXISTS credit_spread DOUBLE PRECISION;
ALTER TABLE macro ADD COLUMN IF NOT EXISTS hmm_regime VARCHAR(20);
ALTER TABLE macro ADD COLUMN IF NOT EXISTS hmm_regime_confidence DOUBLE PRECISION
    CHECK (hmm_regime_confidence BETWEEN 0.0 AND 1.0);

CREATE INDEX IF NOT EXISTS idx_macro_hmm_regime
    ON macro (hmm_regime, timestamp DESC)
    WHERE hmm_regime IS NOT NULL;

COMMENT ON COLUMN macro.hmm_regime IS 'Hidden Markov Model market regime label';
COMMENT ON COLUMN macro.hmm_regime_confidence IS 'Posterior probability of current HMM regime, range [0,1]';
COMMENT ON COLUMN macro.dxy_trend IS 'US Dollar Index trend: strengthening | weakening | neutral';
COMMENT ON COLUMN macro.real_rate_proxy IS 'Approximated real interest rate (nominal - inflation expectation)';
COMMENT ON COLUMN macro.credit_spread IS 'High-yield minus investment-grade spread in basis points';

COMMIT;
