"""Step-by-step assessment for one application.

Each step is something a reviewer can show: the inputs, the checks, the
banking ratios, the model score, the risk band, and a short written result.
"""

from __future__ import annotations

import pandas as pd

from src.config import CREDIT_SCORE_MAX, CREDIT_SCORE_MIN, DTI_MAX
from src.features import add_features
from src.predict import score_record
from src.train import coefficient_table


def build_assessment(record: dict, artifact: dict) -> dict:
    """Score one application and return the ordered steps behind that score."""
    checks = _checks(record)
    steps = [
        _input_step(record),
        _check_step(checks),
    ]
    if not all(item["passed"] for item in checks):
        return {"ok": False, "steps": steps, "pd": None, "risk_class": None}

    featured = add_features(pd.DataFrame([record])).iloc[0]
    scored = score_record(record, artifact)
    steps.extend(
        [
            _feature_step(record, featured),
            _model_step(artifact),
            _risk_step(scored),
            _agent_step(scored),
        ]
    )
    return {
        "ok": True,
        "steps": steps,
        "pd": scored["pd"],
        "risk_class": scored["risk_class"],
        "model_version": scored["model_version"],
        "business_id": scored["business_id"],
    }


def _checks(record: dict) -> list[dict]:
    rules = [
        ("Annual revenue is positive", record["annual_revenue"] > 0),
        ("Total assets are positive", record["total_assets"] > 0),
        ("Total debt is positive", record["total_debt"] > 0),
        ("Loan amount is positive", record["loan_amount"] > 0),
        (
            f"Credit score is between {CREDIT_SCORE_MIN} and {CREDIT_SCORE_MAX}",
            CREDIT_SCORE_MIN <= int(record["credit_score"]) <= CREDIT_SCORE_MAX,
        ),
        ("DTI is between 0 and 5", 0 <= record["dti"] <= DTI_MAX),
        ("Delinquencies are zero or more", int(record["delinq_12m"]) >= 0),
        ("Years in business are zero or more", int(record["years_in_business"]) >= 0),
    ]
    return [{"label": label, "passed": bool(passed)} for label, passed in rules]


def _input_step(record: dict) -> dict:
    return {
        "id": "input",
        "title": "Application received",
        "summary": f"Business {record['business_id']} submitted a loan application.",
        "items": [
            {"label": "Annual revenue", "value": _money(record["annual_revenue"])},
            {"label": "Profit", "value": _money(record["profit"])},
            {"label": "Total debt", "value": _money(record["total_debt"])},
            {"label": "Total assets", "value": _money(record["total_assets"])},
            {"label": "Cash flow", "value": _money(record["cash_flow"])},
            {"label": "Loan amount", "value": _money(record["loan_amount"])},
            {"label": "Credit score", "value": str(int(record["credit_score"]))},
            {"label": "DTI", "value": f"{record['dti']:.4f}"},
            {"label": "Delinquencies, 12 months", "value": str(int(record["delinq_12m"]))},
            {"label": "Years in business", "value": str(int(record["years_in_business"]))},
        ],
    }


def _check_step(checks: list[dict]) -> dict:
    passed = sum(item["passed"] for item in checks)
    return {
        "id": "checks",
        "title": "Data checks",
        "summary": f"{passed} of {len(checks)} checks passed.",
        "items": [
            {"label": item["label"], "value": "Pass" if item["passed"] else "Fail"}
            for item in checks
        ],
    }


def _feature_step(record: dict, featured: pd.Series) -> dict:
    ratios = [
        (
            "Debt to asset",
            "total debt / total assets",
            record["total_debt"],
            record["total_assets"],
            featured["debt_to_asset"],
        ),
        (
            "Loan to revenue",
            "loan amount / annual revenue",
            record["loan_amount"],
            record["annual_revenue"],
            featured["loan_to_revenue"],
        ),
        (
            "Profit margin",
            "profit / annual revenue",
            record["profit"],
            record["annual_revenue"],
            featured["profit_margin"],
        ),
        (
            "Cash-flow coverage",
            "cash flow / total debt",
            record["cash_flow"],
            record["total_debt"],
            featured["cash_flow_coverage"],
        ),
    ]
    return {
        "id": "features",
        "title": "Banking ratios",
        "summary": "Four ratios are calculated from the application before scoring.",
        "items": [
            {
                "label": name,
                "value": f"{float(value):.4f}",
                "detail": f"{formula}: {_money(left)} / {_money(right)}",
            }
            for name, formula, left, right, value in ratios
        ],
    }


def _model_step(artifact: dict) -> dict:
    drivers = []
    for row in coefficient_table(artifact).head(4).itertuples(index=False):
        effect = "raises default risk" if row.coefficient > 0 else "lowers default risk"
        drivers.append(
            {
                "label": row.feature.replace("_", " "),
                "value": effect,
                "detail": f"Coefficient {row.coefficient:+.2f} per standard deviation",
            }
        )
    return {
        "id": "model",
        "title": "Model score",
        "summary": (
            f"{artifact['model_version']} is a logistic regression. "
            "Inputs are standardized on the training book, then converted to a probability."
        ),
        "items": drivers,
    }


def _risk_step(scored: dict) -> dict:
    percent = float(scored["pd"]) * 100
    return {
        "id": "risk",
        "title": "Risk class",
        "summary": (
            f"Probability of default is {percent:.2f}%, which falls in {scored['risk_class']}."
        ),
        "items": [
            {"label": "Probability of default", "value": f"{scored['pd']:.4f}"},
            {"label": "Risk class", "value": scored["risk_class"]},
            {"label": "Below 5%", "value": "Low Risk"},
            {"label": "5% up to 10%", "value": "Moderate Risk"},
            {"label": "10% through 20%", "value": "High Risk"},
            {"label": "Above 20%", "value": "Very High Risk"},
        ],
    }


def _agent_step(scored: dict) -> dict:
    percent = float(scored["pd"]) * 100
    return {
        "id": "agent",
        "title": "Assessment",
        "summary": (
            f"{scored['business_id']} has a probability of default of {percent:.2f}% "
            f"({scored['risk_class']}), model {scored['model_version']}."
        ),
        "items": [],
    }


def _money(value: float) -> str:
    return f"{value:,.2f}"
