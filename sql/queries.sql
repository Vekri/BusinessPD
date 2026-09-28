-- Reporting queries for the loaded PD book.
-- Each block starts with "-- name: <query_name>".
-- Latest-score filters use the highest prediction_id per business so a later
-- rescore does not double-count the portfolio.

-- name: latest_scores
SELECT
    b.business_id,
    b.business_name,
    p.model_version,
    p.pd,
    p.risk_class,
    p.prediction_date
FROM business_customer b
JOIN model_prediction p
    ON p.business_id = b.business_id
WHERE p.prediction_id = (
    SELECT MAX(m.prediction_id)
    FROM model_prediction m
    WHERE m.business_id = b.business_id
)
ORDER BY b.business_id;

-- name: risk_summary
SELECT
    p.risk_class,
    COUNT(*) AS businesses,
    ROUND(AVG(p.pd), 4) AS average_pd,
    ROUND(SUM(a.loan_amount), 2) AS loan_amount
FROM business_customer b
JOIN loan_application a
    ON a.business_id = b.business_id
JOIN model_prediction p
    ON p.business_id = b.business_id
WHERE p.prediction_id = (
    SELECT MAX(m.prediction_id)
    FROM model_prediction m
    WHERE m.business_id = b.business_id
)
AND a.application_id = (
    SELECT MAX(l.application_id)
    FROM loan_application l
    WHERE l.business_id = b.business_id
)
GROUP BY p.risk_class
ORDER BY MIN(p.pd);

-- name: pd_above_15
SELECT
    b.business_id,
    b.business_name,
    b.industry,
    p.model_version,
    p.pd,
    p.risk_class,
    a.loan_amount
FROM business_customer b
JOIN loan_application a
    ON a.business_id = b.business_id
JOIN model_prediction p
    ON p.business_id = b.business_id
WHERE p.prediction_id = (
    SELECT MAX(m.prediction_id)
    FROM model_prediction m
    WHERE m.business_id = b.business_id
)
AND a.application_id = (
    SELECT MAX(l.application_id)
    FROM loan_application l
    WHERE l.business_id = b.business_id
)
AND p.pd > 0.15
ORDER BY p.pd DESC;
