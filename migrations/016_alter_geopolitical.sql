BEGIN;

ALTER TABLE geopolitical ADD COLUMN IF NOT EXISTS geo_risk_index DOUBLE PRECISION
    CHECK (geo_risk_index BETWEEN 0.0 AND 100.0);
ALTER TABLE geopolitical ADD COLUMN IF NOT EXISTS sanctions_activity_score DOUBLE PRECISION
    CHECK (sanctions_activity_score >= 0.0);
ALTER TABLE geopolitical ADD COLUMN IF NOT EXISTS trade_policy_direction DOUBLE PRECISION
    CHECK (trade_policy_direction BETWEEN -1.0 AND 1.0);

CREATE INDEX IF NOT EXISTS idx_geopolitical_category_ts
    ON geopolitical (category, timestamp DESC);

COMMENT ON COLUMN geopolitical.geo_risk_index IS 'Composite geopolitical risk score [0-100], higher = more risk';
COMMENT ON COLUMN geopolitical.sanctions_activity_score IS 'Sanctions event activity intensity, >= 0';
COMMENT ON COLUMN geopolitical.trade_policy_direction IS 'Trade policy direction [-1,1], negative = protectionist, positive = free trade';

COMMIT;
