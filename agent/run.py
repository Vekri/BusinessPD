"""Choose one PD tool for a question and turn the result into a sentence.

This router calls the same API routes as the MCP server. It does not write SQL.
A hosted language model is not configured, so the choice of tool is made here.
"""

from __future__ import annotations

import re

from src.api_client import api_request

_BUSINESS_ID = re.compile(r"\bB\d{4,}\b", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")


def answer(question: str) -> str:
    """Return a plain-language answer for one question."""
    kind, arguments = plan(question)
    if kind == "prediction":
        return _format_prediction(api_request("GET", f"/prediction/{arguments['business_id']}"))
    if kind == "business":
        return _format_business(api_request("GET", f"/business/{arguments['business_id']}"))
    if kind == "portfolio":
        payload = api_request(
            "GET",
            "/portfolio",
            params={"min_pd": arguments["min_pd"], "limit": arguments["limit"]},
        )
        return _format_portfolio(payload)
    if kind == "search":
        payload = api_request(
            "GET",
            "/businesses",
            params={"name": arguments["name"], "industry": arguments["industry"], "limit": 5},
        )
        return _format_search(payload, arguments)
    if kind == "model":
        return _format_model(api_request("GET", "/model"))
    return (
        "Ask for a business id, a PD cutoff such as 15%, a name or industry search, "
        "or the model information."
    )


def plan(question: str) -> tuple[str, dict]:
    """Pick the tool and arguments for a question."""
    text = question.strip()
    lower = text.lower()
    found = _BUSINESS_ID.search(text)
    business_id = found.group(0).upper() if found else ""
    asks_for_list = any(word in lower for word in ("above", "over", "greater than", "higher than"))
    asks_for_score = any(word in lower for word in ("calculat", "probability", "pd", "score", "default"))

    if asks_for_list and ("pd" in lower or "default" in lower or "%" in lower or "percent" in lower):
        return "portfolio", {"min_pd": _threshold(lower), "limit": 5}
    if business_id and asks_for_score:
        return "prediction", {"business_id": business_id}
    if business_id:
        return "business", {"business_id": business_id}
    if any(word in lower for word in ("search", "find", "named", "industry")):
        return "search", _search_terms(text)
    if "model" in lower or "auc" in lower or "gini" in lower:
        return "model", {}
    return "unknown", {}


def _threshold(lower: str) -> float:
    match = _PERCENT.search(lower)
    if not match:
        return 0.15
    return float(match.group(1)) / 100


def _search_terms(text: str) -> dict:
    industry = ""
    name = ""
    for candidate in (
        "healthcare",
        "retail",
        "manufacturing",
        "construction",
        "wholesale",
        "hospitality",
        "transportation",
        "agriculture",
        "technology",
    ):
        if candidate in text.lower():
            industry = candidate
            break
    if "professional" in text.lower():
        industry = "Professional"
    quoted = re.search(r'"([^"]+)"', text)
    if quoted:
        name = quoted.group(1)
    return {"name": name, "industry": industry}


def _format_prediction(row: dict) -> str:
    if "error" in row:
        return str(row["error"])
    percent = float(row["pd"]) * 100
    return (
        f"{row['business_id']} has a probability of default of {percent:.2f}% "
        f"({row['risk_class']}), model {row['model_version']}."
    )


def _format_business(row: dict) -> str:
    if "error" in row:
        return str(row["error"])
    return (
        f"{row['business_id']} is {row['business_name']} in {row['industry']}, "
        f"with credit score {row['credit_score']} and DTI {row['dti']}."
    )


def _format_portfolio(payload: dict) -> str:
    if "error" in payload:
        return str(payload["error"])
    cutoff = float(payload["min_pd"]) * 100
    lines = [f"{payload['count']:,} businesses have a PD above {cutoff:g}%."]
    for row in payload.get("businesses", []):
        lines.append(
            f"{row['business_id']} {row['business_name']}: PD {float(row['pd']):.4f} ({row['risk_class']})"
        )
    if payload["count"] > payload["returned"]:
        lines.append(f"Showing {payload['returned']} of {payload['count']:,}, highest PD first.")
    return "\n".join(lines)


def _format_search(payload: dict, arguments: dict) -> str:
    if "error" in payload:
        return str(payload["error"])
    rows = payload.get("businesses", [])
    label = arguments["industry"] or arguments["name"] or "that search"
    if not rows:
        return f"No businesses matched {label}."
    lines = [f"{payload['count']} businesses matched {label}."]
    for row in rows:
        lines.append(f"{row['business_id']} {row['business_name']} ({row['industry']})")
    return "\n".join(lines)


def _format_model(info: dict) -> str:
    if "error" in info:
        return str(info["error"])
    holdout = info.get("holdout") or {}
    auc = holdout.get("roc_auc")
    auc_text = f" Holdout ROC-AUC is {auc:.3f}." if isinstance(auc, float) else ""
    return (
        f"The active model is {info['model_version']}, a {info['algorithm']} "
        f"with class weight {info['class_weight']}.{auc_text}"
    )
