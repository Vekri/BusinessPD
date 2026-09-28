"""The client walkthrough lists every stage from inputs to the written result."""

from src.assess import build_assessment
from src.data_generator import generate_business_credit
from src.features import add_features
from src.preprocessing import clean_business_credit
from src.train import fit_model

SAMPLE = {
    "business_id": "B1001",
    "annual_revenue": 750000,
    "profit": 85000,
    "total_debt": 300000,
    "total_assets": 850000,
    "credit_score": 680,
    "dti": 0.42,
    "delinq_12m": 1,
    "years_in_business": 6,
    "cash_flow": 110000,
    "loan_amount": 200000,
}


def test_assessment_steps_follow_the_review_order():
    clean = clean_business_credit(generate_business_credit(n_records=300, seed=61)).clean
    artifact = fit_model(add_features(clean))
    result = build_assessment(SAMPLE, artifact)

    assert result["ok"] is True
    assert [step["id"] for step in result["steps"]] == [
        "input",
        "checks",
        "features",
        "model",
        "risk",
        "agent",
    ]
    assert result["steps"][1]["summary"].startswith("8 of 8")
    assert result["steps"][2]["items"][0]["label"] == "Debt to asset"
    assert 0 <= result["pd"] <= 1
    assert result["business_id"] == "B1001"
    assert "B1001" in result["steps"][-1]["summary"]
