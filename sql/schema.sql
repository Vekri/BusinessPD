-- Business PD schema.
-- Customers, the loan application, and the model score are stored separately
-- so a business can be scored again without rewriting its financial record.
-- Re-running this file drops the three tables and recreates them empty.

DROP TABLE IF EXISTS model_prediction;
DROP TABLE IF EXISTS loan_application;
DROP TABLE IF EXISTS business_customer;

CREATE TABLE business_customer (
    business_id VARCHAR(50) PRIMARY KEY,
    business_name VARCHAR(200) NOT NULL,
    industry VARCHAR(100) NOT NULL,
    annual_revenue NUMERIC(18, 2) NOT NULL CHECK (annual_revenue > 0),
    profit NUMERIC(18, 2) NOT NULL,
    total_debt NUMERIC(18, 2) NOT NULL CHECK (total_debt > 0),
    total_assets NUMERIC(18, 2) NOT NULL CHECK (total_assets > 0),
    credit_score INTEGER NOT NULL CHECK (credit_score BETWEEN 300 AND 850),
    dti NUMERIC(10, 4) NOT NULL CHECK (dti >= 0 AND dti <= 5),
    delinq_12m INTEGER NOT NULL CHECK (delinq_12m >= 0),
    years_in_business INTEGER NOT NULL CHECK (years_in_business >= 0),
    cash_flow NUMERIC(18, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE loan_application (
    application_id BIGSERIAL PRIMARY KEY,
    business_id VARCHAR(50) NOT NULL REFERENCES business_customer (business_id),
    loan_amount NUMERIC(18, 2) NOT NULL CHECK (loan_amount > 0),
    application_date DATE NOT NULL DEFAULT CURRENT_DATE,
    status VARCHAR(30) NOT NULL DEFAULT 'submitted'
);

CREATE TABLE model_prediction (
    prediction_id BIGSERIAL PRIMARY KEY,
    business_id VARCHAR(50) NOT NULL REFERENCES business_customer (business_id),
    model_version VARCHAR(50) NOT NULL,
    pd NUMERIC(12, 8) NOT NULL CHECK (pd >= 0 AND pd <= 1),
    risk_class VARCHAR(50) NOT NULL CHECK (
        risk_class IN ('Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk')
    ),
    prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX loan_application_business_id_idx ON loan_application (business_id);
CREATE INDEX model_prediction_business_id_idx ON model_prediction (business_id);
CREATE INDEX model_prediction_prediction_date_idx ON model_prediction (prediction_date);
