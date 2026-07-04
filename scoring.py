"""Simple, explainable scoring for the data-center watchlist."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


CATEGORY_PURE_PLAY = "Pure-play data center REIT/operator"
CATEGORY_DIVERSIFIED = "Diversified real estate / infrastructure with data-center exposure"
CATEGORY_VOLATILE = "Volatile AI/HPC infrastructure / power-secured data-center play"
CATEGORY_INDIRECT = "Asset manager / indirect exposure"

WEIGHTS = {
    "valuation": 0.25,
    "momentum": 0.20,
    "purity": 0.20,
    "ai_demand": 0.20,
    "safety": 0.15,
}

RISK_SCORES = {
    "Low": 90,
    "Medium": 70,
    "High": 40,
    "Very high": 20,
}


def _number(value) -> float | None:
    try:
        output = float(value)
        if math.isnan(output) or math.isinf(output):
            return None
        return output
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _text(row: pd.Series, *columns: str) -> str:
    parts = []
    for column in columns:
        value = row.get(column, "")
        if pd.notna(value):
            parts.append(str(value))
    return " ".join(parts).lower()


def valuation_score(row: pd.Series) -> float:
    pe = _number(row.get("pe"))
    dividend_yield = _number(row.get("dividend_yield"))
    category = row.get("category", "")

    if pe is None or pe <= 0:
        pe_score = 45 if category in {CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED} else 35
    elif pe <= 12:
        pe_score = 95
    elif pe <= 20:
        pe_score = 85
    elif pe <= 35:
        pe_score = 65
    elif pe <= 60:
        pe_score = 45
    else:
        pe_score = 20

    if dividend_yield is None:
        yield_score = 35
    elif dividend_yield >= 5:
        yield_score = 90
    elif dividend_yield >= 3:
        yield_score = 75
    elif dividend_yield >= 1:
        yield_score = 55
    else:
        yield_score = 35

    return _clip((pe_score * 0.70) + (yield_score * 0.30))


def momentum_score(row: pd.Series) -> float:
    returns = [
        _number(row.get("return_1m")),
        _number(row.get("return_3m")),
        _number(row.get("return_6m")),
        _number(row.get("return_12m")),
    ]
    usable = [value for value in returns if value is not None]
    if not usable:
        score = 35
    else:
        weighted = []
        if returns[0] is not None:
            weighted.append(50 + returns[0] * 1.00)
        if returns[1] is not None:
            weighted.append(50 + returns[1] * 0.70)
        if returns[2] is not None:
            weighted.append(50 + returns[2] * 0.35)
        if returns[3] is not None:
            weighted.append(50 + returns[3] * 0.20)
        score = float(np.mean([_clip(value) for value in weighted]))

    rsi = _number(row.get("rsi"))
    six_month_return = _number(row.get("return_6m"))
    if rsi is not None and rsi > 75:
        score -= 20
    if six_month_return is not None and six_month_return > 150:
        score -= 20
    return _clip(score)


def purity_score(row: pd.Series) -> float:
    return _clip(_number(row.get("data_center_purity_score")) or 0)


def ai_demand_score(row: pd.Series) -> float:
    category = row.get("category", "")
    text = _text(row, "notes", "ai_exposure_notes", "known_customers", "risk_notes")

    if category == CATEGORY_VOLATILE:
        score = 75
    elif category == CATEGORY_PURE_PLAY:
        score = 65
    elif category == CATEGORY_DIVERSIFIED:
        score = 50
    else:
        score = 40

    positive_keywords = [
        "ai",
        "hpc",
        "hyperscale",
        "hyperscaler",
        "high-density",
        "cloud",
        "power-secured",
        "campus",
        "data-center",
        "data center",
    ]
    score += min(20, sum(4 for keyword in positive_keywords if keyword in text))

    if "not a data-center owner" in text or "not pure-play" in text:
        score -= 10
    if "indirect" in text:
        score -= 8
    if "unknown" in text or "not disclosed" in text:
        score -= 4
    if "crypto-heavy" in text or "bitcoin mining" in text or "crypto exposure" in text:
        score -= 12

    return _clip(score)


def safety_score(row: pd.Series) -> float:
    risk = str(row.get("risk_level", "Medium"))
    category = row.get("category", "")
    company_type = str(row.get("company_type", "")).lower()
    dividend_yield = _number(row.get("dividend_yield"))
    text = _text(row, "notes", "risk_notes")

    score = RISK_SCORES.get(risk, 55)
    if category == CATEGORY_PURE_PLAY:
        score += 6
    if "reit" in company_type:
        score += 6
    if dividend_yield is not None and dividend_yield >= 2:
        score += 4
    if category == CATEGORY_VOLATILE:
        score -= 18
    if "crypto" in text or "bitcoin mining" in text:
        score -= 8
    if "very high volatility" in text or "extreme volatility" in text:
        score -= 8

    return _clip(score)


def missing_market_data_count(row: pd.Series) -> int:
    required = [
        "price",
        "market_cap",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_12m",
        "rsi",
        "pe",
    ]
    return sum(1 for column in required if _number(row.get(column)) is None)


def score_company(row: pd.Series) -> dict:
    valuation = valuation_score(row)
    momentum = momentum_score(row)
    purity = purity_score(row)
    ai_demand = ai_demand_score(row)
    safety = safety_score(row)

    score = (
        valuation * WEIGHTS["valuation"]
        + momentum * WEIGHTS["momentum"]
        + purity * WEIGHTS["purity"]
        + ai_demand * WEIGHTS["ai_demand"]
        + safety * WEIGHTS["safety"]
    )

    missing_count = missing_market_data_count(row)
    rsi = _number(row.get("rsi"))
    six_month_return = _number(row.get("return_6m"))
    risk = str(row.get("risk_level", ""))
    category = row.get("category", "")
    text = _text(row, "notes", "risk_notes", "ai_exposure_notes")

    score -= min(15, missing_count * 3)
    if rsi is not None and rsi > 75:
        score -= 5
    if six_month_return is not None and six_month_return > 150:
        score -= 8
    if risk == "Very high":
        score -= 6
    if category == CATEGORY_INDIRECT:
        score -= 5
    if "crypto-heavy" in text or "bitcoin mining" in text:
        score -= 6

    final_score = round(_clip(score), 1)
    labels = []

    if missing_count >= 4 or _number(row.get("price")) is None:
        labels.append("Data missing")
    elif final_score >= 75:
        labels.append("Strong candidate")
    elif final_score >= 60:
        labels.append("Interesting")
    else:
        labels.append("Watchlist only")

    if category == CATEGORY_VOLATILE or risk in {"High", "Very high"}:
        labels.append("High risk / speculative")
    if rsi is not None and rsi > 75:
        labels.append("Overbought warning")
    if six_month_return is not None and six_month_return > 150:
        labels.append("May have already exploded")

    return {
        "valuation_score": round(valuation, 1),
        "momentum_score": round(momentum, 1),
        "purity_score_component": round(purity, 1),
        "ai_demand_score": round(ai_demand, 1),
        "safety_score": round(safety, 1),
        "missing_market_data_count": missing_count,
        "final_score": final_score,
        "label": "; ".join(dict.fromkeys(labels)),
    }


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Append scoring columns to a merged watchlist and market-data table."""
    if df.empty:
        return df.copy()

    scored = df.copy()
    score_rows = scored.apply(score_company, axis=1, result_type="expand")
    return pd.concat([scored, score_rows], axis=1)
