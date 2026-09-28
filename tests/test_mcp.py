"""MCP tools call the API and do not build SQL from the question."""

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app

_SPEC = importlib.util.spec_from_file_location(
    "pd_mcp_server",
    Path(__file__).resolve().parents[1] / "mcp" / "server.py",
)
server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(server)


def test_tools_delegate_to_the_api(monkeypatch):
    calls = []

    def fake_request(method, path, json_body=None, params=None):
        calls.append((method, path, json_body, params))
        return {"ok": True, "path": path}

    monkeypatch.setattr(server, "api_request", fake_request)
    assert server.get_business("B00002")["path"] == "/business/B00002"
    assert server.get_latest_prediction("B00002")["path"] == "/prediction/B00002"
    scored = server.calculate_pd(
        business_id="B1001",
        annual_revenue=750000,
        profit=85000,
        total_debt=300000,
        total_assets=850000,
        credit_score=680,
        dti=0.42,
        delinq_12m=1,
        years_in_business=6,
        cash_flow=110000,
        loan_amount=200000,
    )
    assert scored["path"] == "/predict"
    assert calls[2][2]["business_id"] == "B1001"
    assert server.get_portfolio_risk(0.15, 5)["path"] == "/portfolio"
    assert server.search_businesses(industry="Health")["path"] == "/businesses"
    assert server.get_model_information()["path"] == "/model"
    assert [call[0] for call in calls] == ["GET", "GET", "POST", "GET", "GET", "GET"]


def test_api_error_is_returned_not_raised(monkeypatch):
    class Response:
        status_code = 404
        text = "missing"

        def json(self):
            return {"detail": "Business not found: B99999"}

    monkeypatch.setattr("src.api_client.httpx.request", lambda *args, **kwargs: Response())
    result = server.get_business("B99999")
    assert result["status_code"] == 404
    assert "B99999" in result["error"]


def test_portfolio_and_model_routes(monkeypatch):
    monkeypatch.setattr(
        "api.repository.portfolio",
        lambda min_pd, limit: {"min_pd": min_pd, "count": 2, "returned": 1, "businesses": [], "risk_summary": []},
    )
    monkeypatch.setattr("api.repository.search_businesses", lambda name, industry, limit: [])
    artifact = {"model_version": "pd_model_v1", "features": ["dti"], "train_rows": 8000}
    with TestClient(create_app(artifact=artifact)) as client:
        assert client.get("/portfolio", params={"min_pd": 0.15}).json()["count"] == 2
        assert client.get("/portfolio", params={"min_pd": 1.5}).status_code == 400
        info = client.get("/model").json()
        assert info["model_version"] == "pd_model_v1"
        assert info["features"] == ["dti"]
        assert info["algorithm"] == "LogisticRegression"
