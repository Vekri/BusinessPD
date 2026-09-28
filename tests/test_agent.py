"""The agent picks a tool and answers from that tool's payload."""

from agent.run import answer, plan


def test_plan_routes_the_two_example_questions():
    kind, arguments = plan("Calculate the default probability for business B1001.")
    assert kind == "prediction"
    assert arguments == {"business_id": "B1001"}

    kind, arguments = plan("Show me all businesses with PD above 15%.")
    assert kind == "portfolio"
    assert arguments["min_pd"] == 0.15


def test_answer_formats_the_tool_payload(monkeypatch):
    def fake_request(method, path, json_body=None, params=None):
        if path == "/prediction/B1001":
            return {
                "business_id": "B1001",
                "pd": 0.2822,
                "risk_class": "Very High Risk",
                "model_version": "pd_model_v1",
            }
        if path == "/portfolio":
            return {
                "min_pd": params["min_pd"],
                "count": 2,
                "returned": 1,
                "businesses": [
                    {
                        "business_id": "B00005",
                        "business_name": "Pioneer Trade",
                        "pd": 0.95,
                        "risk_class": "Very High Risk",
                    }
                ],
            }
        raise AssertionError(path)

    monkeypatch.setattr("agent.run.api_request", fake_request)
    scored = answer("Calculate the default probability for business B1001.")
    assert "B1001" in scored
    assert "28.22%" in scored
    assert "Very High Risk" in scored

    listed = answer("Show me all businesses with PD above 15%.")
    assert "2 businesses" in listed
    assert "B00005" in listed
