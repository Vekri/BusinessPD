"""End-to-end path from raw rows to a score, then the live API and agent."""

import httpx
import pytest

from agent.run import answer
from src.data_generator import generate_business_credit
from src.evaluate import evaluate_frame
from src.features import add_features
from src.predict import score_record
from src.preprocessing import clean_business_credit
from src.split import split_frame
from src.train import fit_model

RISK_CLASSES = {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}
API_URL = "http://127.0.0.1:8001"


def test_raw_rows_become_a_holdout_score():
    raw = generate_business_credit(n_records=400, seed=51)
    cleaned = clean_business_credit(raw)
    assert cleaned.report.rows_rejected == 0

    featured = add_features(cleaned.clean)
    train, test = split_frame(featured, seed=51)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"]))

    artifact = fit_model(train)
    assert artifact["converged"] is True

    held_out = test.iloc[0].to_dict()
    scored = score_record(held_out, artifact)
    assert scored["business_id"] == held_out["business_id"]
    assert 0 <= scored["pd"] <= 1
    assert scored["risk_class"] in RISK_CLASSES
    assert scored["model_version"] == artifact["model_version"]

    report = evaluate_frame(test, artifact)
    assert report["roc_auc"] > 0.7
    assert report["gini"] == pytest.approx(2 * report["roc_auc"] - 1)


def test_live_api_and_agent_answer_from_the_stored_book():
    try:
        health = httpx.get(f"{API_URL}/health", timeout=3)
    except httpx.HTTPError:
        pytest.skip("PD API is not running")
    if health.status_code != 200:
        pytest.skip("PD API is not running")

    body = health.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["model_version"] == "pd_model_v1"

    business = httpx.get(f"{API_URL}/business/B00002", timeout=10)
    assert business.status_code == 200
    assert business.json()["business_name"] == "Harbor Forge Healthcare"

    scored = answer("Calculate the default probability for business B1001.")
    assert "B1001" in scored
    assert "28.22%" in scored
    assert "Very High Risk" in scored

    portfolio = answer("Show me all businesses with PD above 15%.")
    assert "PD above 15%" in portfolio
    assert "B05787" in portfolio
