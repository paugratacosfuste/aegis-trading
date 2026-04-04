-- SHAP feature importance tracking per model version
CREATE TABLE IF NOT EXISTS feature_importance (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL REFERENCES model_versions(model_id),
    retrain_date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    shap_importance DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'stable',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_imp_model
    ON feature_importance (model_id);

CREATE INDEX IF NOT EXISTS idx_feat_imp_date
    ON feature_importance (retrain_date DESC);
