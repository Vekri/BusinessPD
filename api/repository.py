"""Read and write customers, applications, and scores."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db import get_engine


def ping(engine: Engine | None = None) -> bool:
    engine = engine or get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def save_application(record: dict, scored: dict, engine: Engine | None = None) -> None:
    """Upsert the customer, then append one application and one prediction."""
    engine = engine or get_engine()
    customer = {
        "business_id": record["business_id"],
        "business_name": record.get("business_name") or record["business_id"],
        "industry": record.get("industry") or "Unknown",
        "has_name": record.get("business_name") is not None,
        "has_industry": record.get("industry") is not None,
        "annual_revenue": record["annual_revenue"],
        "profit": record["profit"],
        "total_debt": record["total_debt"],
        "total_assets": record["total_assets"],
        "credit_score": int(record["credit_score"]),
        "dti": record["dti"],
        "delinq_12m": int(record["delinq_12m"]),
        "years_in_business": int(record["years_in_business"]),
        "cash_flow": record["cash_flow"],
        "loan_amount": record["loan_amount"],
        "model_version": scored["model_version"],
        "pd": scored["pd"],
        "risk_class": scored["risk_class"],
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO business_customer (
                    business_id, business_name, industry, annual_revenue, profit,
                    total_debt, total_assets, credit_score, dti, delinq_12m,
                    years_in_business, cash_flow
                ) VALUES (
                    :business_id, :business_name, :industry, :annual_revenue, :profit,
                    :total_debt, :total_assets, :credit_score, :dti, :delinq_12m,
                    :years_in_business, :cash_flow
                )
                ON CONFLICT (business_id) DO UPDATE SET
                    business_name = CASE
                        WHEN :has_name THEN EXCLUDED.business_name
                        ELSE business_customer.business_name
                    END,
                    industry = CASE
                        WHEN :has_industry THEN EXCLUDED.industry
                        ELSE business_customer.industry
                    END,
                    annual_revenue = EXCLUDED.annual_revenue,
                    profit = EXCLUDED.profit,
                    total_debt = EXCLUDED.total_debt,
                    total_assets = EXCLUDED.total_assets,
                    credit_score = EXCLUDED.credit_score,
                    dti = EXCLUDED.dti,
                    delinq_12m = EXCLUDED.delinq_12m,
                    years_in_business = EXCLUDED.years_in_business,
                    cash_flow = EXCLUDED.cash_flow
                """
            ),
            customer,
        )
        connection.execute(
            text(
                """
                INSERT INTO loan_application (business_id, loan_amount, status)
                VALUES (:business_id, :loan_amount, 'scored')
                """
            ),
            customer,
        )
        connection.execute(
            text(
                """
                INSERT INTO model_prediction (business_id, model_version, pd, risk_class)
                VALUES (:business_id, :model_version, :pd, :risk_class)
                """
            ),
            customer,
        )


def _numbers(row: dict) -> dict:
    converted = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            converted[key] = value.isoformat(sep=" ")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            converted[key] = float(value)
        else:
            converted[key] = value
    if "pd" in converted and converted["pd"] is not None:
        converted["pd"] = round(float(converted["pd"]), 4)
    return converted


def get_business(business_id: str, engine: Engine | None = None) -> dict | None:
    engine = engine or get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT business_id, business_name, industry, annual_revenue, profit,
                       total_debt, total_assets, credit_score, dti, delinq_12m,
                       years_in_business, cash_flow
                FROM business_customer
                WHERE business_id = :business_id
                """
            ),
            {"business_id": business_id},
        ).mappings().first()
    return _numbers(dict(row)) if row else None


def get_latest_prediction(business_id: str, engine: Engine | None = None) -> dict | None:
    engine = engine or get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT business_id, model_version, pd, risk_class, prediction_date
                FROM model_prediction
                WHERE business_id = :business_id
                ORDER BY prediction_id DESC
                LIMIT 1
                """
            ),
            {"business_id": business_id},
        ).mappings().first()
    return _numbers(dict(row)) if row else None


def _bounded_limit(limit: int) -> int:
    return max(1, min(int(limit), 50))


def search_businesses(
    name: str = "",
    industry: str = "",
    limit: int = 20,
    engine: Engine | None = None,
) -> list[dict]:
    """Case-insensitive partial match on name and industry. Each filter is optional."""
    engine = engine or get_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT business_id, business_name, industry
                FROM business_customer
                WHERE (:name = '' OR business_name ILIKE '%' || :name || '%')
                  AND (:industry = '' OR industry ILIKE '%' || :industry || '%')
                ORDER BY business_id
                LIMIT :limit
                """
            ),
            {
                "name": name.strip(),
                "industry": industry.strip(),
                "limit": _bounded_limit(limit),
            },
        ).mappings().all()
    return [_numbers(dict(row)) for row in rows]


def portfolio(min_pd: float = 0.15, limit: int = 20, engine: Engine | None = None) -> dict:
    """Latest-score portfolio view. The business list is capped; count is the full total."""
    engine = engine or get_engine()
    latest = """
        p.prediction_id = (
            SELECT MAX(m.prediction_id)
            FROM model_prediction m
            WHERE m.business_id = p.business_id
        )
    """
    params = {"min_pd": min_pd, "limit": _bounded_limit(limit)}
    with engine.connect() as connection:
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM model_prediction p WHERE {latest} AND p.pd > :min_pd"),
            params,
        ).scalar_one()
        summary = connection.execute(
            text(
                f"""
                SELECT p.risk_class, COUNT(*) AS businesses, ROUND(AVG(p.pd), 4) AS average_pd
                FROM model_prediction p
                WHERE {latest}
                GROUP BY p.risk_class
                ORDER BY MIN(p.pd)
                """
            )
        ).mappings().all()
        rows = connection.execute(
            text(
                f"""
                SELECT b.business_id, b.business_name, b.industry,
                       p.pd, p.risk_class, p.model_version
                FROM business_customer b
                JOIN model_prediction p ON p.business_id = b.business_id
                WHERE {latest} AND p.pd > :min_pd
                ORDER BY p.pd DESC
                LIMIT :limit
                """
            ),
            params,
        ).mappings().all()
    businesses = [_numbers(dict(row)) for row in rows]
    return {
        "min_pd": min_pd,
        "count": int(count),
        "returned": len(businesses),
        "risk_summary": [_numbers(dict(row)) for row in summary],
        "businesses": businesses,
    }
