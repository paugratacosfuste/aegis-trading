BEGIN;

ALTER TABLE leader_signals ADD COLUMN IF NOT EXISTS direction DOUBLE PRECISION
    CHECK (direction BETWEEN -1.0 AND 1.0);
ALTER TABLE leader_signals ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION
    CHECK (confidence BETWEEN 0.0 AND 1.0);
ALTER TABLE leader_signals ADD COLUMN IF NOT EXISTS market_impact_estimate DOUBLE PRECISION
    CHECK (market_impact_estimate BETWEEN -1.0 AND 1.0);

CREATE INDEX IF NOT EXISTS idx_leader_signals_direction_ts
    ON leader_signals (direction, timestamp DESC)
    WHERE direction IS NOT NULL;

COMMENT ON COLUMN leader_signals.direction IS 'Estimated directional signal [-1,1], positive = bullish';
COMMENT ON COLUMN leader_signals.confidence IS 'Signal confidence [0,1]';
COMMENT ON COLUMN leader_signals.market_impact_estimate IS 'Estimated directional market impact [-1,1], positive = bullish';

COMMIT;
