-- Monthly automated retrospective reports
CREATE TABLE IF NOT EXISTS retrospective_reports (
    id BIGSERIAL PRIMARY KEY,
    month VARCHAR(7) NOT NULL UNIQUE,
    report JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
