-- Healthcare Fraud Detection Database Schema - FIXED VERSION
-- This file initializes the database structure

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== MAIN TABLES ====================

-- Claims table: Store all healthcare claims
CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR(50) PRIMARY KEY,
    provider_id VARCHAR(50) NOT NULL,
    patient_id VARCHAR(50) NOT NULL,
    claim_amount DECIMAL(12,2) NOT NULL,
    deductible_paid DECIMAL(12,2) DEFAULT 0,
    patient_age INTEGER,
    hospital_stay_days INTEGER DEFAULT 0,
    claim_duration INTEGER DEFAULT 1,
    gender INTEGER DEFAULT 1, -- 1=Male, 2=Female
    chronic_alzheimer BOOLEAN DEFAULT FALSE,
    chronic_heartfailure BOOLEAN DEFAULT FALSE,
    chronic_kidney BOOLEAN DEFAULT FALSE,
    chronic_cancer BOOLEAN DEFAULT FALSE,
    chronic_diabetes BOOLEAN DEFAULT FALSE,
    claim_status VARCHAR(20) DEFAULT 'PENDING',
    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- ADD THESE COLUMNS FOR TOP 10 FEATURES SUPPORT
    patient_total_claims INTEGER DEFAULT 1,
    patient_provider_count INTEGER DEFAULT 1,
    race_risk_group INTEGER DEFAULT 1,
    op_annual_amount DECIMAL(12,2) DEFAULT 0
);
-- Create indexes for claims table
CREATE INDEX IF NOT EXISTS idx_claims_provider ON claims(provider_id);
CREATE INDEX IF NOT EXISTS idx_claims_patient ON claims(patient_id);
CREATE INDEX IF NOT EXISTS idx_claims_submission_date ON claims(submission_date);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(claim_status);
CREATE INDEX IF NOT EXISTS idx_claims_amount ON claims(claim_amount);

-- Fraud predictions table: Store ML model predictions
CREATE TABLE IF NOT EXISTS fraud_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id VARCHAR(50) NOT NULL,
    fraud_probability DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    risk_level VARCHAR(20) NOT NULL,
    risk_score INTEGER, -- 1-100 scale
    risk_factors JSONB, -- Store risk factors as JSON
    model_version VARCHAR(20) DEFAULT 'v1.0',
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    investigated BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE
);

-- Create indexes for fraud_predictions table
CREATE INDEX IF NOT EXISTS idx_predictions_claim ON fraud_predictions(claim_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON fraud_predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON fraud_predictions(fraud_probability);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON fraud_predictions(prediction_timestamp);

-- Investigations table: Track fraud investigations
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id VARCHAR(50) NOT NULL,
    prediction_id UUID,
    priority_level VARCHAR(20) NOT NULL, -- CRITICAL, HIGH, MEDIUM, LOW
    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, IN_PROGRESS, CLOSED
    assigned_investigator VARCHAR(100),
    findings TEXT,
    investigation_outcome VARCHAR(30), -- FRAUD_CONFIRMED, FRAUD_DENIED, INSUFFICIENT_EVIDENCE
    amount_recovered DECIMAL(12,2) DEFAULT 0,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completion_date TIMESTAMP,
    
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES fraud_predictions(prediction_id)
);

-- Create indexes for investigations table
CREATE INDEX IF NOT EXISTS idx_investigations_claim ON investigations(claim_id);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);
CREATE INDEX IF NOT EXISTS idx_investigations_priority ON investigations(priority_level);
CREATE INDEX IF NOT EXISTS idx_investigations_outcome ON investigations(investigation_outcome);

-- ==================== SYSTEM SETTINGS ====================

-- System settings for configurable thresholds and parameters
CREATE TABLE IF NOT EXISTS system_settings (
    setting_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    setting_name VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(20) DEFAULT 'STRING', -- STRING, INTEGER, DECIMAL, BOOLEAN
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'system'
);

-- Insert default system settings
INSERT INTO system_settings (setting_name, setting_value, setting_type, description) VALUES
('fraud_threshold_conservative', '0.90', 'DECIMAL', 'Conservative fraud detection threshold (90%)'),
('fraud_threshold_balanced', '0.80', 'DECIMAL', 'Balanced fraud detection threshold (80%)'),
('fraud_threshold_aggressive', '0.70', 'DECIMAL', 'Aggressive fraud detection threshold (70%)'),
('max_investigation_capacity', '100', 'INTEGER', 'Maximum number of investigations per day'),
('auto_investigation_threshold', '0.95', 'DECIMAL', 'Auto-create investigation above this threshold')
ON CONFLICT (setting_name) DO NOTHING;

-- ==================== FIXED VIEWS ====================

-- Provider statistics view for quick analytics - FIXED
CREATE OR REPLACE VIEW provider_statistics AS
SELECT 
    provider_id,
    COUNT(*) as total_claims,
    AVG(claim_amount) as avg_claim_amount,
    SUM(claim_amount) as total_claim_amount,
    COUNT(CASE WHEN fp.risk_level IN ('Critical Risk', 'High Risk') THEN 1 END) as critical_cases,
    AVG(fp.fraud_probability) as avg_fraud_prob,
    COUNT(CASE WHEN i.investigation_outcome = 'FRAUD_CONFIRMED' THEN 1 END) as confirmed_fraud_cases
FROM claims c
LEFT JOIN fraud_predictions fp ON c.claim_id = fp.claim_id
LEFT JOIN investigations i ON c.claim_id = i.claim_id
GROUP BY provider_id
HAVING COUNT(*) > 0;

-- Daily fraud summary view - FIXED
CREATE OR REPLACE VIEW daily_fraud_summary AS
SELECT 
    DATE(c.submission_date) as report_date,
    COUNT(*) as total_claims,
    COUNT(CASE WHEN fp.risk_level = 'Critical Risk' THEN 1 END) as critical_risk_claims,
    COUNT(CASE WHEN fp.risk_level = 'High Risk' THEN 1 END) as high_risk_claims,
    COUNT(CASE WHEN fp.risk_level = 'Medium Risk' THEN 1 END) as medium_risk_claims,
    COUNT(CASE WHEN fp.risk_level = 'Low Risk' THEN 1 END) as low_risk_claims,
    AVG(c.claim_amount) as avg_claim_amount,
    SUM(c.claim_amount) as total_claim_amount,
    AVG(fp.fraud_probability) as avg_fraud_probability
FROM claims c
LEFT JOIN fraud_predictions fp ON c.claim_id = fp.claim_id
GROUP BY DATE(c.submission_date)
ORDER BY report_date DESC;

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp trigger to claims table
DROP TRIGGER IF EXISTS update_claims_timestamp ON claims;
CREATE TRIGGER update_claims_timestamp
    BEFORE UPDATE ON claims
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- Grant permissions to the fraud_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO fraud_user;