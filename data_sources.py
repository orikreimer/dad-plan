"""Market data helpers for the Streamlit MVP.

The dashboard uses yfinance because it is easy to run locally. It is good
enough for an MVP research screen, but it should not be treated as an
institutional data source.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}


MARKET_DATA_COLUMNS = [
    "ticker",
    "currency",
    "sector",
    "industry",
    "price",
    "market_cap",
    "enterprise_value",
    "pe",
    "price_to_book",
    "dividend_yield",
    "beta",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "distance_from_52w_high",
    "distance_from_52w_low",
    "analyst_target_price",
    "analyst_recommendation",
    "total_revenue",
    "revenue_growth",
    "profit_margins",
    "debt_to_equity",
    "free_cashflow",
    "ebitda",
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "rsi",
    "ma_50",
    "ma_200",
    "distance_50dma",
    "distance_200dma",
    "data_status",
    "data_error",
]


def safe_float(value) -> float:
    """Convert values from APIs to float, returning NaN when unavailable."""
    try:
        if value is None:
            return np.nan
        if isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA", "NONE"}:
            return np.nan
        output = float(value)
        if math.isinf(output):
            return np.nan
        return output
    except (TypeError, ValueError):
        return np.nan


def _has_value(value) -> bool:
    return not pd.isna(safe_float(value))


def _fast_info_value(fast_info, key: str):
    """Read yfinance fast_info whether it behaves like a dict or object."""
    try:
        value = fast_info.get(key)
        if value is not None:
            return value
    except Exception:
        pass

    try:
        return getattr(fast_info, key)
    except Exception:
        return None


def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    """Calculate a simple RSI from adjusted close prices."""
    close = close.dropna()
    if len(close) <= period:
        return np.nan

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    last_loss = avg_loss.iloc[-1]
    last_gain = avg_gain.iloc[-1]
    if pd.isna(last_loss) or pd.isna(last_gain):
        return np.nan
    if last_loss == 0:
        return 100.0

    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _return_pct(close: pd.Series, trading_days: int) -> float:
    close = close.dropna()
    if len(close) <= trading_days:
        return np.nan
    start = safe_float(close.iloc[-trading_days - 1])
    end = safe_float(close.iloc[-1])
    if not _has_value(start) or not _has_value(end) or start == 0:
        return np.nan
    return float((end / start - 1) * 100)


def _moving_average(close: pd.Series, window: int) -> float:
    close = close.dropna()
    if len(close) < window:
        return np.nan
    return float(close.rolling(window=window).mean().iloc[-1])


def _distance_from_ma(price: float, moving_average: float) -> float:
    if not _has_value(price) or not _has_value(moving_average) or moving_average == 0:
        return np.nan
    return float((price / moving_average - 1) * 100)


def _distance_from_reference(price: float, reference: float) -> float:
    if not _has_value(price) or not _has_value(reference) or reference == 0:
        return np.nan
    return float((price / reference - 1) * 100)


def _percent_from_yfinance(value) -> float:
    number = safe_float(value)
    if not _has_value(number):
        return np.nan
    return number * 100 if abs(number) <= 1 else number


def _blank_market_row(ticker: str, status: str = "N/A", error: str = "") -> dict:
    row = {column: np.nan for column in MARKET_DATA_COLUMNS}
    row["ticker"] = ticker
    row["currency"] = "N/A"
    row["sector"] = "N/A"
    row["industry"] = "N/A"
    row["analyst_recommendation"] = "N/A"
    row["data_status"] = status
    row["data_error"] = error
    return row


def fetch_market_data(ticker: str) -> dict:
    """Fetch one ticker from yfinance without letting failures crash the app."""
    row = _blank_market_row(ticker)
    errors: list[str] = []

    try:
        stock = yf.Ticker(ticker)
    except Exception as exc:
        row["data_status"] = "N/A"
        row["data_error"] = f"Ticker init failed: {exc}"
        return row

    hist = pd.DataFrame()
    try:
        hist = stock.history(period="18mo", interval="1d", auto_adjust=True, actions=False)
    except Exception as exc:
        errors.append(f"history: {exc}")

    close = pd.Series(dtype=float)
    if not hist.empty and "Close" in hist.columns:
        close = hist["Close"].dropna()

    if not close.empty:
        row["price"] = safe_float(close.iloc[-1])
        row["return_1m"] = _return_pct(close, TRADING_DAYS["1m"])
        row["return_3m"] = _return_pct(close, TRADING_DAYS["3m"])
        row["return_6m"] = _return_pct(close, TRADING_DAYS["6m"])
        row["return_12m"] = _return_pct(close, TRADING_DAYS["12m"])
        row["rsi"] = calculate_rsi(close)
        row["ma_50"] = _moving_average(close, 50)
        row["ma_200"] = _moving_average(close, 200)
        row["distance_50dma"] = _distance_from_ma(row["price"], row["ma_50"])
        row["distance_200dma"] = _distance_from_ma(row["price"], row["ma_200"])

    try:
        fast_info = stock.fast_info
        fast_price = safe_float(
            _fast_info_value(fast_info, "last_price")
            or _fast_info_value(fast_info, "lastPrice")
            or _fast_info_value(fast_info, "regular_market_price")
        )
        if _has_value(fast_price):
            row["price"] = fast_price

        market_cap = safe_float(
            _fast_info_value(fast_info, "market_cap")
            or _fast_info_value(fast_info, "marketCap")
        )
        if _has_value(market_cap):
            row["market_cap"] = market_cap
    except Exception as exc:
        errors.append(f"fast_info: {exc}")

    info = {}
    try:
        info = stock.info or {}
    except Exception as exc:
        errors.append(f"info: {exc}")

    if info:
        row["currency"] = info.get("currency") or row["currency"]
        row["sector"] = info.get("sector") or row["sector"]
        row["industry"] = info.get("industry") or row["industry"]

        if not _has_value(row["price"]):
            row["price"] = safe_float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
        if not _has_value(row["market_cap"]):
            row["market_cap"] = safe_float(info.get("marketCap"))

        row["enterprise_value"] = safe_float(info.get("enterpriseValue"))
        row["pe"] = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        row["price_to_book"] = safe_float(info.get("priceToBook"))
        row["beta"] = safe_float(info.get("beta"))
        row["fifty_two_week_high"] = safe_float(info.get("fiftyTwoWeekHigh"))
        row["fifty_two_week_low"] = safe_float(info.get("fiftyTwoWeekLow"))
        row["analyst_target_price"] = safe_float(info.get("targetMeanPrice"))
        row["analyst_recommendation"] = (
            info.get("recommendationKey")
            or info.get("recommendationMean")
            or row["analyst_recommendation"]
        )
        row["total_revenue"] = safe_float(info.get("totalRevenue"))
        row["revenue_growth"] = _percent_from_yfinance(info.get("revenueGrowth"))
        row["profit_margins"] = _percent_from_yfinance(info.get("profitMargins"))
        row["debt_to_equity"] = safe_float(info.get("debtToEquity"))
        row["free_cashflow"] = safe_float(info.get("freeCashflow"))
        row["ebitda"] = safe_float(info.get("ebitda"))

        dividend_yield = safe_float(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
        if _has_value(dividend_yield):
            # yfinance usually returns dividendYield as a decimal, for example 0.032.
            row["dividend_yield"] = dividend_yield * 100 if dividend_yield <= 1 else dividend_yield

    if _has_value(row["price"]):
        row["distance_from_52w_high"] = _distance_from_reference(
            row["price"], row["fifty_two_week_high"]
        )
        row["distance_from_52w_low"] = _distance_from_reference(
            row["price"], row["fifty_two_week_low"]
        )

    if _has_value(row["price"]):
        row["data_status"] = "OK"
    else:
        row["data_status"] = "N/A"
        errors.append("No usable price returned")

    row["data_error"] = "; ".join(errors)
    return row


def fetch_market_data_for_watchlist(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch all market rows and always return the expected schema."""
    rows = []
    for ticker in tickers:
        rows.append(fetch_market_data(str(ticker).strip()))

    if not rows:
        return pd.DataFrame(columns=MARKET_DATA_COLUMNS)

    df = pd.DataFrame(rows)
    for column in MARKET_DATA_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[MARKET_DATA_COLUMNS]
