"""Quiver Quant integration helpers.

Quiver uses an authorization token. The first MVP used QUIVER_API_KEY as the
environment variable name, so this module still supports it as a fallback.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


QUIVER_BASE_URL = "https://api.quiverquant.com"
NOT_CONNECTED = "Quiver not connected"
SUBSCRIPTION_REQUIRED = "Quiver connected; dataset requires subscription upgrade"
REQUEST_FAILED = "Quiver request failed"


def get_quiver_auth_token() -> str:
    """Load the Quiver authorization token from .env or the environment.

    QUIVER_API_KEY remains supported because the first MVP used that name.
    """
    load_dotenv()
    return (
        os.getenv("QUIVER_AUTH_TOKEN", "").strip()
        or os.getenv("QUIVER_API_KEY", "").strip()
    )


def get_quiver_api_key() -> str:
    """Backward-compatible alias for older code."""
    return get_quiver_auth_token()


def is_quiver_connected() -> bool:
    return bool(get_quiver_auth_token())


def get_quiver_access_status() -> dict[str, Any]:
    """Check whether the current token can access premium datasets."""
    return call_quiver_endpoint("/beta/auth/premium")


def call_quiver_endpoint(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generic Quiver GET helper."""
    auth_token = get_quiver_auth_token()
    if not auth_token:
        return {"connected": False, "status": NOT_CONNECTED, "data": None}

    url = f"{QUIVER_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Token {auth_token}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, params=params or {}, timeout=15)
        if response.status_code == 403:
            return {
                "connected": True,
                "status": SUBSCRIPTION_REQUIRED,
                "data": None,
                "status_code": response.status_code,
            }
        if response.status_code == 404:
            return {
                "connected": True,
                "status": "Quiver endpoint not found",
                "data": None,
                "status_code": response.status_code,
            }
        response.raise_for_status()
        return {
            "connected": True,
            "status": "OK",
            "data": response.json(),
            "status_code": response.status_code,
        }
    except requests.RequestException as exc:
        return {"connected": True, "status": f"{REQUEST_FAILED}: {exc}", "data": None}


def _latest_ticker_signal(signal_name: str, ticker: str, endpoint: str) -> dict[str, Any]:
    result = call_quiver_endpoint(endpoint.format(ticker=ticker.upper()))
    return {
        "ticker": ticker,
        "signal": signal_name,
        "status": result["status"],
        "score": _score_from_result(result),
        "rows": _row_count(result.get("data")),
        "data": result.get("data"),
    }


def get_congress_trading_signal(ticker: str) -> dict[str, Any]:
    return _latest_ticker_signal(
        "Congress trading",
        ticker,
        "/beta/historical/congresstrading/{ticker}",
    )


def get_insider_trading_signal(ticker: str) -> dict[str, Any]:
    # Quiver publishes a live insider feed. The MVP does not yet normalize
    # that full feed by ticker, so this returns connection/status metadata.
    result = call_quiver_endpoint("/beta/live/insiders")
    return {
        "ticker": ticker,
        "signal": "Insider trading",
        "status": result["status"],
        "score": _score_from_result(result),
        "rows": _row_count(result.get("data")),
        "data": result.get("data"),
    }


def get_lobbying_signal(ticker: str) -> dict[str, Any]:
    return _latest_ticker_signal(
        "Lobbying",
        ticker,
        "/beta/historical/lobbying/{ticker}",
    )


def get_government_contracts_signal(ticker: str) -> dict[str, Any]:
    return _latest_ticker_signal(
        "Government contracts",
        ticker,
        "/beta/historical/govcontracts/{ticker}",
    )


def get_news_signal(ticker: str) -> dict[str, Any]:
    result = call_quiver_endpoint("/beta/live/quivernews")
    return {
        "ticker": ticker,
        "signal": "Quiver news",
        "status": result["status"],
        "score": _score_from_result(result),
        "rows": _row_count(result.get("data")),
        "data": result.get("data"),
    }


def get_combined_alternative_signal(ticker: str) -> str:
    """Return a simple display string for the dashboard table."""
    if not is_quiver_connected():
        return NOT_CONNECTED

    signals = [
        get_congress_trading_signal(ticker),
        get_lobbying_signal(ticker),
        get_government_contracts_signal(ticker),
    ]
    statuses = {signal["status"] for signal in signals}
    if statuses == {SUBSCRIPTION_REQUIRED}:
        return SUBSCRIPTION_REQUIRED
    if "OK" in statuses:
        total_rows = sum(signal.get("rows") or 0 for signal in signals)
        return f"Quiver connected; {total_rows} rows across checked datasets"
    return "; ".join(sorted(statuses))


def get_quiver_status_summary(ticker: str) -> dict[str, Any]:
    """Return detailed signal statuses for the company detail view."""
    return {
        "connected": is_quiver_connected(),
        "congress": get_congress_trading_signal(ticker),
        "insider": get_insider_trading_signal(ticker),
        "lobbying": get_lobbying_signal(ticker),
        "government_contracts": get_government_contracts_signal(ticker),
        "news": get_news_signal(ticker),
    }


def _row_count(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return len(value)
        return 1 if data else 0
    return 0


def _score_from_result(result: dict[str, Any]) -> int | None:
    if result.get("status") != "OK":
        return None
    rows = _row_count(result.get("data"))
    if rows <= 0:
        return 0
    return min(100, 35 + rows * 5)
