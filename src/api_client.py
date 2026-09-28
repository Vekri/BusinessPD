"""HTTP client for the PD API. MCP tools and the agent both use this."""

from __future__ import annotations

import os

import httpx


def api_base() -> str:
    return os.environ.get("PD_API_URL", "http://127.0.0.1:8001").rstrip("/")


def api_request(
    method: str,
    path: str,
    json_body: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Call one API route and return JSON, or an error object."""
    try:
        response = httpx.request(
            method,
            f"{api_base()}{path}",
            json=json_body,
            params=params,
            timeout=60,
        )
    except httpx.HTTPError as exc:
        return {"error": f"API unavailable: {exc}"}
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        return {"error": detail, "status_code": response.status_code}
    return response.json()
