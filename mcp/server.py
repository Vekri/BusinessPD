"""MCP tools for the business PD service.

The tools call the HTTP API. They do not accept or run SQL.

    python mcp/server.py

Cursor launches that command over stdio. PD_API_URL defaults to http://127.0.0.1:8001.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ORIGINAL_PATH = sys.path[:]
_blocked = {_SCRIPT_DIR, _PROJECT_ROOT}
_kept = []
for _entry in sys.path:
    _candidate = Path.cwd() if not _entry else Path(_entry)
    try:
        _resolved = _candidate.resolve()
    except OSError:
        _kept.append(_entry)
        continue
    if _resolved not in _blocked:
        _kept.append(_entry)
sys.path[:] = _kept

from mcp.server.mcpserver import MCPServer

sys.path[:] = _ORIGINAL_PATH
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.api_client import api_request

mcp = MCPServer("business-pd")


@mcp.tool()
def get_business(business_id: str) -> dict:
    """Return the stored customer record for one business id, such as B00002."""
    return api_request("GET", f"/business/{business_id}")


@mcp.tool()
def calculate_pd(
    business_id: str,
    annual_revenue: float,
    profit: float,
    total_debt: float,
    total_assets: float,
    credit_score: int,
    dti: float,
    delinq_12m: int,
    years_in_business: int,
    cash_flow: float,
    loan_amount: float,
    business_name: str = "",
    industry: str = "",
) -> dict:
    """Score one application and store the customer, loan, and prediction."""
    body = {
        "business_id": business_id,
        "annual_revenue": annual_revenue,
        "profit": profit,
        "total_debt": total_debt,
        "total_assets": total_assets,
        "credit_score": credit_score,
        "dti": dti,
        "delinq_12m": delinq_12m,
        "years_in_business": years_in_business,
        "cash_flow": cash_flow,
        "loan_amount": loan_amount,
    }
    if business_name:
        body["business_name"] = business_name
    if industry:
        body["industry"] = industry
    return api_request("POST", "/predict", json_body=body)


@mcp.tool()
def get_latest_prediction(business_id: str) -> dict:
    """Return the newest stored probability of default for a business id."""
    return api_request("GET", f"/prediction/{business_id}")


@mcp.tool()
def get_portfolio_risk(min_pd: float = 0.15, limit: int = 20) -> dict:
    """List businesses whose latest PD is above min_pd. The list is capped; count is the full total."""
    return api_request("GET", "/portfolio", params={"min_pd": min_pd, "limit": limit})


@mcp.tool()
def search_businesses(name: str = "", industry: str = "", limit: int = 20) -> dict:
    """Find businesses by a partial name and/or industry. Both filters are optional."""
    return api_request(
        "GET",
        "/businesses",
        params={"name": name, "industry": industry, "limit": limit},
    )


@mcp.tool()
def get_model_information() -> dict:
    """Return the active model version, features, risk bands, and holdout metrics."""
    return api_request("GET", "/model")


if __name__ == "__main__":
    mcp.run()
