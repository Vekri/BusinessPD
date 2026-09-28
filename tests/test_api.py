"""API routes score a business and read the stored result."""

from fastapi.testclient import TestClient

from api.main import PredictRequest, create_app
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


def _client(monkeypatch) -> TestClient:
    clean = clean_business_credit(generate_business_credit(n_records=300, seed=41)).clean
    artifact = fit_model(add_features(clean))
    saved: dict[str, dict] = {}

    def save_application(record, scored):
        saved["business"] = {
            "business_id": record["business_id"],
            "business_name": record.get("business_name") or record["business_id"],
            "annual_revenue": record["annual_revenue"],
        }
        saved["prediction"] = scored

    monkeypatch.setattr("api.repository.save_application", save_application)
    monkeypatch.setattr("api.repository.get_business", lambda business_id: saved.get("business") if saved.get("business", {}).get("business_id") == business_id else None)
    monkeypatch.setattr(
        "api.repository.get_latest_prediction",
        lambda business_id: saved.get("prediction") if saved.get("prediction", {}).get("business_id") == business_id else None,
    )
    monkeypatch.setattr("api.repository.ping", lambda: True)
    return create_app(artifact=artifact)


def test_health_and_predict_round_trip(monkeypatch):
    with TestClient(_client(monkeypatch)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["database"] == "ok"

        scored = client.post("/predict", json=SAMPLE)
        assert scored.status_code == 200
        body = scored.json()
        assert body["business_id"] == "B1001"
        assert 0 < body["pd"] < 1
        assert body["risk_class"]
        assert body["model_version"]

        fetched = client.get("/prediction/B1001")
        assert fetched.status_code == 200
        assert fetched.json()["pd"] == body["pd"]
        assert client.get("/business/B1001").json()["business_name"] == "B1001"
        assert client.get("/business/MISSING").status_code == 404
        assert client.get("/prediction/MISSING").status_code == 404


def test_credit_score_outside_range_is_rejected():
    payload = dict(SAMPLE)
    payload["credit_score"] = 900
    assert PredictRequest.model_validate(SAMPLE)
    with TestClient(create_app(artifact={"model_version": "test", "pipeline": None})) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
