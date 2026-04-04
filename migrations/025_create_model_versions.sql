-- Trained model version tracking for walk-forward retraining
CREATE TABLE IF NOT EXISTS model_versions (
    model_id VARCHAR(64) PRIMARY KEY,
    model_type VARCHAR(40) NOT NULL,
    version VARCHAR(30) NOT NULL,
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    val_start DATE NOT NULL,
    val_end DATE NOT NULL,
    train_samples INTEGER NOT NULL,
    val_samples INTEGER NOT NULL,
    val_auc DOUBLE PRECISION NOT NULL,
    previous_auc DOUBLE PRECISION,
    accepted BOOLEAN NOT NULL,
    feature_names JSONB NOT NULL,
    model_path VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_versions_type
    ON model_versions (model_type, created_at DESC);
