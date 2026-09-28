"""Named report SQL is present and can be parsed without a database."""

from src.report import QUERIES_PATH, load_queries


def test_report_queries_are_named():
    queries = load_queries(QUERIES_PATH)
    assert set(queries) == {"latest_scores", "risk_summary", "pd_above_15"}
    assert "GROUP BY p.risk_class" in queries["risk_summary"]
    assert "p.pd > 0.15" in queries["pd_above_15"]
    assert "MAX(m.prediction_id)" in queries["latest_scores"]
