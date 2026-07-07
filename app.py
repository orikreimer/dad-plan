from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data_sources import MARKET_DATA_COLUMNS, fetch_market_data_for_watchlist
from quiver import (
    SUBSCRIPTION_REQUIRED,
    get_combined_alternative_signal,
    get_quiver_access_status,
    get_quiver_status_summary,
    is_quiver_connected,
)
from report_export import build_pdf_report
from scoring import (
    CATEGORY_DIVERSIFIED,
    CATEGORY_INDIRECT,
    CATEGORY_PURE_PLAY,
    CATEGORY_VOLATILE,
    add_scores,
)


WATCHLIST_PATH = "watchlist.csv"
PITCH_RESEARCH_PATH = "pitch_research.csv"
PLAIN_LANGUAGE_PATH = "plain_language_research.csv"
VERIFIED_DATA_PATH = "verified_company_data.csv"
QUIVER_NO_USEFUL_SIGNAL = "Quiver connected; no useful signal yet"

PITCH_COLUMNS = [
    "pitch_summary",
    "why_it_could_win",
    "expansion_or_leasing_evidence",
    "customer_or_demand_evidence",
    "power_capacity_notes",
    "why_better_than_peers",
    "main_red_flags",
    "what_to_verify_next",
    "pitch_verdict",
    "evidence_quality",
    "evidence_date",
    "evidence_links",
]

PLAIN_LANGUAGE_COLUMNS = [
    "simple_answer",
    "why_good",
    "why_risky",
    "future_story",
    "why_better_or_worse",
    "what_to_check",
    "best_for",
    "bottom_line",
    "confidence_level",
]

VERIFIED_DATA_COLUMNS = [
    "verified_as_of",
    "data_quality",
    "portfolio_facts",
    "leasing_facts",
    "customer_facts",
    "capacity_power_facts",
    "expansion_facts",
    "verified_source_links",
]

FINAL_FIVE = [
    {
        "rank": 1,
        "ticker": "DLR",
        "company": "Digital Realty Trust",
        "exchange": "NYSE, USA",
        "role": "Direct global data-center REIT",
        "risk": "Medium",
        "checked_price": "$174.68",
        "analyst_view": "S&P Global consensus: Buy",
        "analyst_target": "$219.59",
        "buy_zone": "$175-$185",
        "what_they_do": "Digital Realty owns and operates large data-center campuses used by cloud, enterprise, and AI-heavy customers.",
        "why": "This is one of the clearest public-market ways to own global data-center real estate with meaningful AI and hyperscale demand exposure, without moving into very speculative names.",
        "who_said": "S&P Global's analyst poll showed a Buy consensus and a $219.59 average target. BMO had Buy with a $220 target, Bernstein had Buy with $232, and Truist had Buy with $225. Ben McMillan of IDX Advisors highlighted DLR as an AI-infrastructure name with long leases and relatively protected near-term cash flows.",
        "dad_note": "This is the first pick if the goal is direct data-center exposure without taking extreme speculative risk.",
        "sources": [
            ("StockAnalysis / S&P Global DLR target", "https://stockanalysis.com/stocks/dlr/forecast/"),
            ("Business Insider / Ben McMillan", "https://www.businessinsider.com/stocks-to-buy-ai-trade-hyperscaler-spending-pullback-amzn-dlr-2026-7"),
        ],
    },
    {
        "rank": 2,
        "ticker": "EQIX",
        "company": "Equinix",
        "exchange": "Nasdaq, USA",
        "role": "High-quality global data-center platform",
        "risk": "Low-medium",
        "checked_price": "$1,005.38",
        "analyst_view": "S&P Global consensus: Buy",
        "analyst_target": "$1,199",
        "buy_zone": "$950-$1,030",
        "what_they_do": "Equinix operates a global network of data centers and interconnection hubs where companies connect cloud, networks, and infrastructure.",
        "why": "This is the quality name in the group: many customers, global scale, and less dependence on one AI customer or one major lease.",
        "who_said": "S&P Global's analyst poll showed a Buy consensus and a $1,199 average target. Citi had Buy with a $1,260 target, Bernstein had Buy with $1,222, and Stifel had Buy with $1,250. UBS kept a Buy rating with a $1,035 target after Equinix's analyst day.",
        "dad_note": "This is the quality holding. It is not cheap, but it fits a patient investor who wants less drama.",
        "sources": [
            ("StockAnalysis / S&P Global EQIX target", "https://stockanalysis.com/stocks/eqix/forecast/"),
            ("Investopedia / UBS commentary", "https://www.investopedia.com/equinix-stock-leads-s-and-p-decliners-as-investors-digest-growth-targets-11761647"),
            ("Business Insider / Ben McMillan", "https://www.businessinsider.com/stocks-to-buy-ai-trade-hyperscaler-spending-pullback-amzn-dlr-2026-7"),
        ],
    },
    {
        "rank": 3,
        "ticker": "NXT.AX",
        "company": "NEXTDC",
        "exchange": "ASX, Australia",
        "role": "Australian pure-play data-center growth stock",
        "risk": "Medium-high",
        "checked_price": "A$13.80",
        "analyst_view": "S&P Global consensus: Strong Buy",
        "analyst_target": "A$20.40",
        "buy_zone": "A$14-A$16",
        "what_they_do": "NEXTDC builds and operates data centers in Australia and Asia-Pacific for cloud, enterprise, government, and AI workloads.",
        "why": "This gives focused non-US exposure to data-center growth. It has more upside potential, but also more execution, funding, and construction risk.",
        "who_said": "S&P Global's analyst poll showed a Strong Buy consensus and an A$20.40 average target. Citi had Buy with A$19.10, RBC had Buy with A$22.00, and Canaccord had Buy with A$22.55.",
        "dad_note": "This is a good growth candidate, but it is riskier than DLR and EQIX because it depends more on development and funding.",
        "sources": [
            ("StockAnalysis / S&P Global NXT target", "https://stockanalysis.com/quote/asx/NXT/forecast/"),
            ("NEXTDC investor reports", "https://www.nextdc.com/investor-centre"),
        ],
    },
    {
        "rank": 4,
        "ticker": "BIP / BIP.UN",
        "company": "Brookfield Infrastructure Partners",
        "exchange": "NYSE + TSX, Canada",
        "role": "Diversified infrastructure with data exposure",
        "risk": "Medium",
        "checked_price": "$36.96 USD",
        "analyst_view": "S&P Global consensus: Buy",
        "analyst_target": "$44.18 USD",
        "buy_zone": "BIP under $38 USD",
        "what_they_do": "Brookfield Infrastructure owns essential infrastructure across utilities, transport, energy, and data infrastructure such as towers, fiber, and data centers.",
        "why": "This is not a pure data-center stock, but it is more diversified. It can benefit from AI and digital infrastructure demand without depending only on data centers.",
        "who_said": "S&P Global's analyst poll showed a Buy consensus and a $44.18 average target. CIBC had Buy with $45, Morgan Stanley had Buy with $46, TD Cowen had Buy with $57, Scotiabank had Buy with $44, and BMO had Buy with $44. Brookfield itself says AI is increasing demand for digital infrastructure.",
        "dad_note": "This is the choice if the goal is more diversified infrastructure and income, not a pure data-center stock.",
        "sources": [
            ("StockAnalysis / S&P Global BIP target", "https://stockanalysis.com/stocks/bip/forecast/"),
            ("Brookfield Infrastructure official overview", "https://bip.brookfield.com/"),
        ],
    },
    {
        "rank": 5,
        "ticker": "GMG.AX",
        "company": "Goodman Group",
        "exchange": "ASX, Australia",
        "role": "Industrial property platform with data-center pipeline",
        "risk": "Medium",
        "checked_price": "A$30.68",
        "analyst_view": "S&P Global consensus: Strong Buy",
        "analyst_target": "A$34.66",
        "buy_zone": "A$30-A$32",
        "what_they_do": "Goodman owns, develops, and manages logistics and industrial property, and is increasingly using power-enabled land for data-center development.",
        "why": "This is not a pure data-center company, but it is a high-quality real-estate operator with meaningful data-center optionality.",
        "who_said": "S&P Global's analyst poll showed a Strong Buy consensus and an A$34.66 average target. Morgan Stanley had Buy with A$36.15, Citi had Buy with A$40.00, Bell Potter had Buy with A$35.50, and Jefferies had Buy with A$34.13.",
        "dad_note": "This is a good company, but less pure. It should only be considered at a reasonable price because the upside to the average target is smaller than NEXTDC.",
        "sources": [
            ("StockAnalysis / S&P Global GMG target", "https://stockanalysis.com/quote/asx/GMG/forecast/"),
            ("Goodman data centres", "https://www.goodman.com/our-properties/data-centres"),
        ],
    },
]


def file_version(path: str) -> int:
    """Return a small cache key so edited CSV files refresh in Streamlit."""
    try:
        return Path(path).stat().st_mtime_ns
    except FileNotFoundError:
        return 0


# TODO: Add FMP API for valuation ratios.
# TODO: Add Alpha Vantage API as a backup price/technical source.
# TODO: Add Finnhub API for fundamentals and market data.
# TODO: Add SEC filing parser for REIT metrics and customer disclosures.
# TODO: Add automatic news monitoring for OpenAI, Oracle, Microsoft, Meta, CoreWeave, Stargate, hyperscaler leases, MW capacity, and pre-leasing.
# TODO: Add manual customer relationship database.
# TODO: Add charts and a database instead of CSV.
# TODO: Add scheduled daily refresh and optional login later.
# TODO: Add better PDF formatting, FFO/AFFO metrics, tenant concentration analysis, and power capacity / MW tracking.


st.set_page_config(
    page_title="Data Center Real Estate Watchlist",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #f7f8fa;
        --panel: #ffffff;
        --line: #d9dee7;
        --text: #172033;
        --muted: #5b6472;
        --accent: #2f5f8f;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, p, span, label, div {
        letter-spacing: 0 !important;
    }

    h1 {
        color: var(--text);
        font-size: 2.1rem !important;
        line-height: 1.15 !important;
    }

    h2, h3 {
        color: var(--text);
    }

    .hero {
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: none;
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }

    .hero-title {
        color: var(--text);
        font-size: 2rem;
        font-weight: 760;
        margin: 0 0 0.35rem 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        max-width: 860px;
        margin: 0;
    }

    .hero-badge {
        white-space: nowrap;
        color: var(--muted);
        background: #f3f5f8;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.45rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .dashboard-card {
        min-height: 126px;
        border-radius: 8px;
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        background: var(--panel);
        padding: 0.95rem;
        box-shadow: none;
        overflow: hidden;
    }

    .card-label {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.2;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        font-weight: 700;
    }

    .card-value {
        color: var(--text);
        font-size: 1.35rem;
        line-height: 1.15;
        font-weight: 780;
        margin-bottom: 0.4rem;
        overflow-wrap: anywhere;
    }

    .card-subtitle {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.3;
    }

    .sparkline {
        display: none;
    }

    .panel {
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: none;
    }

    .panel-title {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 760;
        margin-bottom: 0.75rem;
    }

    .simple-note {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .hebrew-note {
        direction: rtl;
        text-align: right;
        color: var(--text);
        background: #f3f5f8;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.65;
        font-size: 0.96rem;
    }

    .hebrew-note strong {
        color: var(--text);
    }

    .hebrew-note ul {
        margin: 0.45rem 1.2rem 0 0;
        padding: 0;
    }

    .hebrew-note li {
        margin: 0.2rem 0;
    }

    .warning-line {
        color: var(--text);
        background: #f8f4ea;
        border: 1px solid #e5d8b6;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-top: 0.55rem;
        font-size: 0.9rem;
    }

    .good-line {
        color: var(--text);
        background: #eef7f1;
        border: 1px solid #cfe4d5;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-top: 0.55rem;
        font-size: 0.9rem;
    }

    .stSelectbox, .stMultiSelect, .stSlider, .stCheckbox {
        color: var(--text);
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #ffffff;
        border-color: var(--line);
        color: var(--text);
        border-radius: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: #ffffff;
        box-shadow: none;
    }

    .stDownloadButton button, .stButton button {
        background: var(--accent);
        color: white;
        border: 0;
        border-radius: 8px;
        font-weight: 750;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f3f5f8;
        border-radius: 8px;
        color: var(--muted);
        padding: 0.45rem 0.8rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--text);
        background: #e6edf5;
    }

    @media (max-width: 900px) {
        .hero-top {
            display: block;
        }
        .hero-badge {
            display: inline-block;
            margin-top: 0.8rem;
        }
        .hero-title {
            font-size: 1.55rem;
        }
        .card-value {
            font-size: 1.12rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _is_number(value) -> bool:
    try:
        return pd.notna(value) and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_number(value, decimals: int = 1) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def fmt_price(value) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.2f}"


def fmt_percent(value) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.1f}%"


def fmt_market_cap(value) -> str:
    if not _is_number(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


def fmt_text(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def fmt_money(value) -> str:
    if not _is_number(value):
        return "N/A"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000_000:
        return f"{sign}${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"
    return f"{sign}${value:,.0f}"


def series_or_na(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    """Return a column or an aligned fallback Series when cached data is older."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def row_value(row: pd.Series, column: str, default=np.nan):
    """Read a row value safely when cached app data is missing newer columns."""
    try:
        return row.get(column, default)
    except AttributeError:
        return default


def display_name(row: pd.Series | None) -> str:
    if row is None or row.empty:
        return "N/A"
    return f"{row.get('company_name', 'N/A')} ({row.get('ticker', 'N/A')})"


def short_category(category: str) -> str:
    if category == CATEGORY_PURE_PLAY:
        return "Pure data-center"
    if category == CATEGORY_DIVERSIFIED:
        return "Diversified infrastructure"
    if category == CATEGORY_VOLATILE:
        return "Speculative AI/HPC"
    if category == CATEGORY_INDIRECT:
        return "Indirect asset manager"
    return category or "Unknown"


def best_row(df: pd.DataFrame, sort_column: str = "final_score", ascending: bool = False) -> pd.Series | None:
    if df.empty or sort_column not in df.columns:
        return None
    usable = df[df[sort_column].notna()].copy()
    if usable.empty:
        return None
    return usable.sort_values(sort_column, ascending=ascending).iloc[0]


@st.cache_data(ttl=3600)
def load_watchlist(csv_version: int) -> pd.DataFrame:
    del csv_version
    df = pd.read_csv(WATCHLIST_PATH, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.strip()
    return df


@st.cache_data(ttl=3600)
def load_pitch_research(csv_version: int) -> pd.DataFrame:
    del csv_version
    try:
        df = pd.read_csv(PITCH_RESEARCH_PATH, dtype={"ticker": str})
    except FileNotFoundError:
        df = pd.DataFrame(columns=["ticker", *PITCH_COLUMNS])

    df["ticker"] = df["ticker"].astype(str).str.strip()
    for column in PITCH_COLUMNS:
        if column not in df.columns:
            df[column] = "Not researched yet."
        df[column] = df[column].fillna("Not researched yet.")
    return df[["ticker", *PITCH_COLUMNS]]


@st.cache_data(ttl=3600)
def load_plain_language_research(csv_version: int) -> pd.DataFrame:
    del csv_version
    try:
        df = pd.read_csv(PLAIN_LANGUAGE_PATH, dtype={"ticker": str})
    except FileNotFoundError:
        df = pd.DataFrame(columns=["ticker", *PLAIN_LANGUAGE_COLUMNS])

    df["ticker"] = df["ticker"].astype(str).str.strip()
    for column in PLAIN_LANGUAGE_COLUMNS:
        if column not in df.columns:
            df[column] = "Not researched yet."
        df[column] = df[column].fillna("Not researched yet.")
    return df[["ticker", *PLAIN_LANGUAGE_COLUMNS]]


@st.cache_data(ttl=3600)
def load_verified_company_data(csv_version: int) -> pd.DataFrame:
    del csv_version
    try:
        df = pd.read_csv(VERIFIED_DATA_PATH, dtype={"ticker": str})
    except FileNotFoundError:
        df = pd.DataFrame(columns=["ticker", *VERIFIED_DATA_COLUMNS])

    if "source_links" in df.columns and "verified_source_links" not in df.columns:
        df = df.rename(columns={"source_links": "verified_source_links"})

    df["ticker"] = df["ticker"].astype(str).str.strip()
    for column in VERIFIED_DATA_COLUMNS:
        if column not in df.columns:
            df[column] = "Not available in checked public sources."
        df[column] = df[column].fillna("Not available in checked public sources.")
    return df[["ticker", *VERIFIED_DATA_COLUMNS]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_data_for_watchlist(tickers)


@st.cache_data(ttl=3600, show_spinner=False)
def load_quiver_signals(tickers: tuple[str, ...]) -> pd.DataFrame:
    if not is_quiver_connected():
        return pd.DataFrame(
            {
                "ticker": tickers,
                "alternative_data_signal": ["Quiver not connected" for _ in tickers],
            }
        )

    access_status = get_quiver_access_status()
    if access_status.get("status") == SUBSCRIPTION_REQUIRED:
        return pd.DataFrame(
            {
                "ticker": tickers,
                "alternative_data_signal": [QUIVER_NO_USEFUL_SIGNAL for _ in tickers],
            }
        )

    return pd.DataFrame(
        {
            "ticker": tickers,
            "alternative_data_signal": [get_combined_alternative_signal(ticker) for ticker in tickers],
        }
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_quiver_detail(ticker: str) -> dict:
    return get_quiver_status_summary(ticker)


def prepare_data() -> pd.DataFrame:
    watchlist = load_watchlist(file_version(WATCHLIST_PATH))
    pitch_research = load_pitch_research(file_version(PITCH_RESEARCH_PATH))
    plain_research = load_plain_language_research(file_version(PLAIN_LANGUAGE_PATH))
    verified_data = load_verified_company_data(file_version(VERIFIED_DATA_PATH))
    tickers = tuple(watchlist["ticker"].dropna().astype(str).tolist())

    with st.spinner("Fetching latest market data..."):
        market_data = load_market_data(tickers)

    quiver_signals = load_quiver_signals(tickers)
    merged = (
        watchlist.merge(pitch_research, on="ticker", how="left")
        .merge(plain_research, on="ticker", how="left")
        .merge(verified_data, on="ticker", how="left")
        .merge(market_data, on="ticker", how="left")
        .merge(quiver_signals, on="ticker", how="left")
    )
    for column in PITCH_COLUMNS:
        merged[column] = merged[column].fillna("Not researched yet.")
    for column in PLAIN_LANGUAGE_COLUMNS:
        merged[column] = merged[column].fillna("Not researched yet.")
    for column in VERIFIED_DATA_COLUMNS:
        merged[column] = merged[column].fillna("Not available in checked public sources.")
    for column in MARKET_DATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    merged["alternative_data_signal"] = merged["alternative_data_signal"].fillna("Quiver not connected")
    return add_scores(merged)


def render_data_source_note(df: pd.DataFrame) -> None:
    yfinance_ok = int((df["data_status"] == "OK").sum()) if "data_status" in df.columns else 0
    total = len(df)
    quiver_values = set(df.get("alternative_data_signal", pd.Series(dtype=str)).dropna().astype(str))
    if QUIVER_NO_USEFUL_SIGNAL in quiver_values:
        quiver_note = "Quiver is connected, but it is not adding a useful signal yet. The stock prices and ratios still come from yfinance."
    elif "Quiver not connected" in quiver_values:
        quiver_note = "Quiver is not connected."
    else:
        quiver_note = "Quiver alternative data is available for the rows that show it."

    st.markdown(
        f"""
        <div class="simple-note">
            Stock prices, market caps, valuation fields, returns, RSI, moving averages, beta,
            52-week range, revenue growth, margins, and analyst fields are from yfinance/Yahoo when available
            ({yfinance_ok}/{total} tickers loaded). {escape(quiver_note)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, value: str, subtitle: str = "", accent: str = "blue") -> None:
    del accent
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-label">{escape(title)}</div>
            <div class="card-value">{escape(value)}</div>
            <div class="card-subtitle">{escape(subtitle)}</div>
            <div class="sparkline"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_start(title: str) -> None:
    st.markdown(f'<div class="panel-title">{escape(title)}</div>', unsafe_allow_html=True)


def render_hebrew_note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="hebrew-note">
            <strong>{escape(title)}</strong><br>
            {escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hebrew_bullets(title: str, items: list[str]) -> None:
    bullet_html = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="hebrew-note">
            <strong>{escape(title)}</strong>
            <ul>{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dad_guide() -> None:
    with st.expander("הסבר פשוט בעברית - איך לקרוא את המערכת", expanded=True):
        render_hebrew_bullets(
            "מה המערכת עושה",
            [
                "המסך הזה לא אומר מה לקנות. הוא מסדר רשימת חברות שקשורות לדאטה סנטרים, נדלן דיגיטלי, חשמל, AI ותשתיות.",
                "המטרה היא לעזור להבין מי נראית יציבה יותר, מי ספקולטיבית יותר, ומה צריך לבדוק לפני שמדברים על מניה.",
                "הציון הוא כלי סינון ראשוני בלבד. החלטה אמיתית צריכה בדיקה נוספת בדוחות, מצגות חברה ונתונים עדכניים.",
            ],
        )
        render_hebrew_bullets(
            "מילון קצר",
            [
                "Score: ציון כללי מ-0 עד 100. גבוה יותר אומר שהמניה מתאימה יותר לנושא לפי המודל, לא שהיא בטוח טובה.",
                "Risk: רמת סיכון. High או Very high אומר שצריך להיזהר במיוחד, גם אם הסיפור נשמע מעניין.",
                "Purity: כמה החברה באמת ממוקדת בדאטה סנטרים. ציון גבוה אומר קשר ישיר יותר לנושא.",
                "AI/HPC: ביקוש שקשור לבינה מלאכותית ומחשוב כבד. זה יכול לתת צמיחה, אבל לפעמים גם הרבה סיכון.",
                "RSI: מדד שמראה אם המניה עלתה חזק מדי בזמן קצר. מעל 75 זו נורת אזהרה לא לרדוף אחרי מחיר חם.",
                "P/E: מכפיל רווח. שימושי בחלק מהמניות, אבל פחות מתאים ל-REITs כי בנדלן משתמשים יותר ב-FFO/AFFO.",
                "Dividend yield: תשואת דיבידנד. הכנסה שנתית יחסית למחיר המניה, אם החברה מחלקת דיבידנד.",
                "52W high gap: כמה המחיר רחוק מהשיא של השנה האחרונה. קרוב מאוד לשיא יכול להיות יקר או חזק, וצריך להבין למה.",
            ],
        )


def render_final_five() -> None:
    with st.container(border=True):
        render_panel_start("Final 5 stocks")
        st.info(
            "This is the narrowed final list from the latest check: USA, Australia, and Canada only. "
            "This does not mean buy today at any price. The Buy area is a price zone for further review, not an automatic buy order."
        )

        final_chart = pd.DataFrame(
            [
                {
                    "Rank": item["rank"],
                    "Stock": f"{item['company']} ({item['ticker']})",
                    "Exchange": item["exchange"],
                    "Exposure type": item["role"],
                    "Risk": item["risk"],
                    "Checked price": item["checked_price"],
                    "Analyst view": item["analyst_view"],
                    "Average target": item["analyst_target"],
                    "Buy area to check": item["buy_zone"],
                }
                for item in FINAL_FIVE
            ]
        )
        st.dataframe(final_chart, hide_index=True, use_container_width=True)
        st.caption("Prices and analyst targets were checked online on 2026-07-07. Re-check price and news before any real decision.")

        st.info(
            "How to read this: Average target is an analyst average, not a promise. "
            "Buy area is a zone where the stock is worth reviewing, not a required purchase price. "
            "If the stock runs far above that area, it is better to wait or re-check the thesis."
        )

        st.markdown("### Explanation for each stock")
        for item in FINAL_FIVE:
            with st.expander(f"{item['rank']}. {item['company']} ({item['ticker']})", expanded=item["rank"] == 1):
                st.markdown(f"**What the company does:** {item['what_they_do']}")
                st.markdown(f"**Why it is in the final 5:** {item['why']}")
                st.markdown(f"**Who said what:** {item['who_said']}")
                st.markdown(f"**Simple note:** {item['dad_note']}")
                st.markdown(f"**Price discipline:** Analyst target is {item['analyst_target']}; buy area to check is {item['buy_zone']}.")
                if item["sources"]:
                    st.markdown("**Sources**")
                    for label, url in item["sources"]:
                        st.markdown(f"- [{label}]({url})")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.container(border=True):
        render_panel_start("Quick filters")
        render_hebrew_note(
            "פילטרים",
            "כאן בוחרים איזה סוג חברות לראות. אם רוצים להתחיל פשוט, להשאיר את הכל פתוח. אם רוצים פחות רעש, לבחור Stable / real-estate style ולהסתיר שמות ספקולטיביים.",
        )
        col1, col2, col3, col4 = st.columns([1.1, 1.2, 1.1, 1])

        view_mode = col1.selectbox(
            "View",
            [
                "All companies",
                "Stable / real-estate style",
                "Pure-play data centers",
                "Volatile AI/HPC only",
                "Indirect asset managers",
            ],
        )
        risk = col2.multiselect(
            "Risk level",
            ["Low", "Medium", "High", "Very high"],
            default=["Low", "Medium", "High", "Very high"],
        )
        country = col3.multiselect(
            "Country",
            sorted(df["country"].dropna().unique()),
            default=sorted(df["country"].dropna().unique()),
        )
        min_score = col4.slider("Minimum score", 0, 100, 0)

        with st.expander("More filters"):
            adv1, adv2, adv3 = st.columns(3)
            category = adv1.multiselect(
                "Category",
                sorted(df["category"].dropna().unique()),
                default=sorted(df["category"].dropna().unique()),
            )
            company_type = adv2.multiselect(
                "Company type",
                sorted(df["company_type"].dropna().unique()),
                default=sorted(df["company_type"].dropna().unique()),
            )
            hide_speculative = adv3.checkbox("Hide speculative names", value=False)
            hide_very_high = adv3.checkbox("Hide very high risk names", value=False)

    filtered = df[
        df["category"].isin(category)
        & df["company_type"].isin(company_type)
        & df["country"].isin(country)
        & df["risk_level"].isin(risk)
        & (df["final_score"] >= min_score)
    ].copy()

    if view_mode == "Stable / real-estate style":
        filtered = filtered[filtered["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED])]
    elif view_mode == "Pure-play data centers":
        filtered = filtered[filtered["category"] == CATEGORY_PURE_PLAY]
    elif view_mode == "Volatile AI/HPC only":
        filtered = filtered[filtered["category"] == CATEGORY_VOLATILE]
    elif view_mode == "Indirect asset managers":
        filtered = filtered[filtered["category"] == CATEGORY_INDIRECT]

    if hide_speculative:
        filtered = filtered[~filtered["label"].astype(str).str.contains("High risk / speculative", na=False)]
    if hide_very_high:
        filtered = filtered[filtered["risk_level"] != "Very high"]

    return filtered


def render_summary_cards(df: pd.DataFrame) -> None:
    ranked = df.sort_values("final_score", ascending=False, na_position="last")
    stable = ranked[ranked["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED])]
    income_style = ranked[
        (ranked["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED]))
        & (ranked["dividend_yield"].fillna(0) > 0)
    ]
    warnings = ranked[
        ranked["label"].astype(str).str.contains("Overbought|exploded|High risk / speculative", case=False, na=False)
    ]

    cards = [
        (
            "Start here",
            best_row(stable),
            lambda row: f"Score {fmt_number(row.get('final_score'))} | Risk {row.get('risk_level', 'N/A')}",
            "blue",
        ),
        (
            "Best overall score",
            best_row(ranked),
            lambda row: f"Score {fmt_number(row.get('final_score'))} | {short_category(row.get('category', ''))}",
            "blue",
        ),
        (
            "Dividend-style name",
            best_row(income_style, "safety_score"),
            lambda row: f"Yield {fmt_percent(row.get('dividend_yield'))} | Risk {row.get('risk_level', 'N/A')}",
            "blue",
        ),
        (
            "Main warning",
            best_row(warnings, "rsi"),
            lambda row: f"{row.get('risk_level', 'N/A')} risk | RSI {fmt_number(row.get('rsi'))}",
            "blue",
        ),
    ]

    columns = st.columns(4)
    for col, (title, row, subtitle_fn, accent) in zip(columns, cards):
        with col:
            if row is None:
                render_card(title, "N/A", "", accent)
            else:
                render_card(title, display_name(row), subtitle_fn(row), accent)

    render_hebrew_note(
        "כרטיסי סיכום",
        "הכרטיסים למעלה הם רק נקודת התחלה. הם לא המלצה לקנייה. עדיף להתחיל מהשם היציב, ואז לפתוח כרטיס חברה ולקרוא את ההסבר.",
    )


def make_display_table(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("final_score", ascending=False, na_position="last")
    return pd.DataFrame(
        {
            "Company": series_or_na(ordered, "company_name", "N/A"),
            "Ticker": series_or_na(ordered, "ticker", "N/A"),
            "Category": series_or_na(ordered, "category", "N/A").map(short_category),
            "Type": series_or_na(ordered, "company_type", "N/A"),
            "Risk": series_or_na(ordered, "risk_level", "N/A"),
            "Country": series_or_na(ordered, "country", "N/A"),
            "Price": series_or_na(ordered, "price").map(fmt_price),
            "Market cap": series_or_na(ordered, "market_cap").map(fmt_market_cap),
            "52W high gap": series_or_na(ordered, "distance_from_52w_high").map(fmt_percent),
            "Beta": series_or_na(ordered, "beta").map(fmt_number),
            "1M": series_or_na(ordered, "return_1m").map(fmt_percent),
            "3M": series_or_na(ordered, "return_3m").map(fmt_percent),
            "6M": series_or_na(ordered, "return_6m").map(fmt_percent),
            "12M": series_or_na(ordered, "return_12m").map(fmt_percent),
            "RSI": series_or_na(ordered, "rsi").map(fmt_number),
            "Yield": series_or_na(ordered, "dividend_yield").map(fmt_percent),
            "P/E": series_or_na(ordered, "pe").map(fmt_number),
            "Revenue growth": series_or_na(ordered, "revenue_growth").map(fmt_percent),
            "Debt/equity": series_or_na(ordered, "debt_to_equity").map(fmt_number),
            "Alt signal": series_or_na(ordered, "alternative_data_signal", "N/A"),
            "Purity": series_or_na(ordered, "data_center_purity_score").map(fmt_number),
            "AI notes": series_or_na(ordered, "ai_exposure_notes", "N/A"),
            "Risk notes": series_or_na(ordered, "risk_notes", "N/A"),
            "Simple verdict": series_or_na(ordered, "bottom_line", "N/A"),
            "Pitch verdict": series_or_na(ordered, "pitch_verdict", "N/A"),
            "Score": series_or_na(ordered, "final_score").map(fmt_number),
            "Label": series_or_na(ordered, "label", "N/A"),
        }
    )


def _value_for_compare(row: pd.Series, column: str, formatter=None) -> str:
    value = row_value(row, column, "N/A")
    if formatter:
        return formatter(value)
    return "N/A" if pd.isna(value) else str(value)


def simple_score_explanation(row: pd.Series) -> list[str]:
    """Hebrew reasons for why the score looks the way it does."""
    reasons = []
    purity = row.get("data_center_purity_score")
    risk = str(row.get("risk_level", "N/A"))
    category = row.get("category", "")
    dividend = row.get("dividend_yield")
    rsi = row.get("rsi")
    six_month = row.get("return_6m")
    revenue_growth = row.get("revenue_growth")
    missing = row.get("missing_market_data_count")

    if _is_number(purity) and float(purity) >= 80:
        reasons.append("קשר חזק לדאטה סנטרים: החברה קרובה מאוד לנושא המרכזי של הרשימה.")
    elif _is_number(purity) and float(purity) <= 35:
        reasons.append("קשר חלש יותר לדאטה סנטרים: זו חשיפה עקיפה יותר לנושא.")

    if risk in {"Low", "Medium"}:
        reasons.append("רמת סיכון נמוכה או בינונית עוזרת לציון.")
    if risk in {"High", "Very high"} or category == CATEGORY_VOLATILE:
        reasons.append("סיכון גבוה מוריד את הציון, גם אם יש פוטנציאל גדול מ-AI או מחשוב כבד.")

    if _is_number(dividend) and float(dividend) > 0:
        reasons.append("דיבידנד עוזר לצד היציבות והערכת השווי.")

    if _is_number(revenue_growth) and float(revenue_growth) > 15:
        reasons.append("הנתונים מראים צמיחת הכנסות חזקה, וזה תומך בסיפור הצמיחה.")
    elif _is_number(revenue_growth) and float(revenue_growth) < 0:
        reasons.append("הנתונים מראים ירידה בצמיחת ההכנסות, ולכן צריך הוכחה חזקה יותר לסיפור.")

    if _is_number(rsi) and float(rsi) > 75:
        reasons.append("RSI מעל 75, ולכן המניה אולי חמה מדי כרגע ולא כדאי לרדוף בלי בדיקה.")
    if _is_number(six_month) and float(six_month) > 150:
        reasons.append("המניה עלתה מעל 150% בחצי שנה, אז ייתכן שחלק גדול מהסיפור כבר במחיר.")
    if _is_number(missing) and float(missing) >= 4:
        reasons.append("חסרים הרבה נתוני שוק, ולכן הציון מקבל פחות ביטחון.")

    if not reasons:
        reasons.append("הציון מאוזן: אין גורם אחד שמסביר אותו לבד.")
    return reasons


def render_peer_comparison(df: pd.DataFrame, row: pd.Series) -> None:
    st.markdown("**Compare against another stock**")
    render_hebrew_note(
        "השוואה מול חברה אחרת",
        "כאן בודקים חברה אחת מול חברה דומה. זה עוזר להבין אם הציון טוב בגלל עסק אמיתי, בגלל מומנטום במחיר, או בגלל שהמתחרה פשוט חלשה יותר.",
    )
    peer_options = [
        f"{peer.ticker} - {peer.company_name}"
        for peer in df.sort_values("ticker").itertuples()
        if peer.ticker != row["ticker"]
    ]
    if not peer_options:
        st.info("No peer available with the current filters.")
        return

    default_peer_index = 0
    same_category = df[
        (df["category"] == row["category"])
        & (df["ticker"] != row["ticker"])
    ].sort_values("final_score", ascending=False)
    if not same_category.empty:
        target = same_category.iloc[0]["ticker"]
        for index, option in enumerate(peer_options):
            if option.startswith(f"{target} - "):
                default_peer_index = index
                break

    selected_peer = st.selectbox("Peer", peer_options, index=default_peer_index)
    peer_ticker = selected_peer.split(" - ", 1)[0]
    peer = df[df["ticker"] == peer_ticker].iloc[0]

    comparison = pd.DataFrame(
        [
            ["Company", display_name(row), display_name(peer)],
            ["Category", short_category(row_value(row, "category", "")), short_category(row_value(peer, "category", ""))],
            ["Risk", fmt_text(row_value(row, "risk_level", "N/A")), fmt_text(row_value(peer, "risk_level", "N/A"))],
            ["Score", fmt_number(row_value(row, "final_score")), fmt_number(row_value(peer, "final_score"))],
            ["Purity", fmt_number(row_value(row, "data_center_purity_score")), fmt_number(row_value(peer, "data_center_purity_score"))],
            ["AI demand score", fmt_number(row_value(row, "ai_demand_score")), fmt_number(row_value(peer, "ai_demand_score"))],
            ["Safety score", fmt_number(row_value(row, "safety_score")), fmt_number(row_value(peer, "safety_score"))],
            ["6M return", fmt_percent(row_value(row, "return_6m")), fmt_percent(row_value(peer, "return_6m"))],
            ["52W high gap", fmt_percent(row_value(row, "distance_from_52w_high")), fmt_percent(row_value(peer, "distance_from_52w_high"))],
            ["RSI", fmt_number(row_value(row, "rsi")), fmt_number(row_value(peer, "rsi"))],
            ["Dividend yield", fmt_percent(row_value(row, "dividend_yield")), fmt_percent(row_value(peer, "dividend_yield"))],
            ["P/E", fmt_number(row_value(row, "pe")), fmt_number(row_value(peer, "pe"))],
            ["Revenue growth", fmt_percent(row_value(row, "revenue_growth")), fmt_percent(row_value(peer, "revenue_growth"))],
            ["Simple bottom line", fmt_text(row_value(row, "bottom_line", "N/A")), fmt_text(row_value(peer, "bottom_line", "N/A"))],
            ["Confidence", fmt_text(row_value(row, "confidence_level", "N/A")), fmt_text(row_value(peer, "confidence_level", "N/A"))],
        ],
        columns=["Question", fmt_text(row_value(row, "ticker", "Selected")), fmt_text(row_value(peer, "ticker", "Peer"))],
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("**Why this one might be better**")
    st.write(row_value(row, "why_better_or_worse", "N/A"))
    st.caption(row_value(row, "why_better_than_peers", "N/A"))
    st.markdown("**Why the peer might still be better**")
    st.write(row_value(peer, "why_better_or_worse", "N/A"))
    st.caption(row_value(peer, "why_better_than_peers", "N/A"))


def render_detail(df: pd.DataFrame) -> None:
    with st.container(border=True):
        render_panel_start("Company detail")
        render_hebrew_note(
            "כרטיס חברה",
            "כאן בוחרים חברה אחת ומקבלים הסבר פשוט: מה היא עושה, למה היא יכולה לעבוד, מה מסוכן בה, אילו עובדות אומתו, ומה אומרים המספרים.",
        )
        if df.empty:
            st.info("No company matches the current filters.")
            return

        options = [f"{row.ticker} - {row.company_name}" for row in df.sort_values("ticker").itertuples()]
        selected = st.selectbox("Pick one company to explain", options)
        ticker = selected.split(" - ", 1)[0]
        row = df[df["ticker"] == ticker].iloc[0]

        headline_cols = st.columns(4)
        with headline_cols[0]:
            render_card("Company", display_name(row), short_category(row_value(row, "category", "")), "blue")
        with headline_cols[1]:
            render_card("Final score", fmt_number(row_value(row, "final_score")), fmt_text(row_value(row, "label", "N/A")), "green")
        with headline_cols[2]:
            render_card("Risk level", fmt_text(row_value(row, "risk_level", "N/A")), f"Purity {fmt_number(row_value(row, 'data_center_purity_score'))}", "amber")
        with headline_cols[3]:
            render_card("Stock move", fmt_percent(row_value(row, "return_6m")), f"6M return | RSI {fmt_number(row_value(row, 'rsi'))}", "violet")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Pitch", "Evidence", "Numbers", "Compare"]
        )
        with tab1:
            render_hebrew_note(
                "Pitch",
                "זה החלק הכי חשוב לקריאה מהירה. הוא עונה בשפה פשוטה: האם החברה מעניינת, למה, מה יכול להשתבש, ומה צריך לבדוק לפני שמציגים אותה.",
            )
            st.markdown("### Simple answer")
            st.info(row_value(row, "simple_answer", "N/A"))

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Why this can be good**")
                st.success(row_value(row, "why_good", "N/A"))
                st.markdown("**What the future depends on**")
                st.write(row_value(row, "future_story", "N/A"))
                st.markdown("**Who this is best for**")
                st.write(row_value(row, "best_for", "N/A"))
            with right:
                st.markdown("**Why this can be bad**")
                st.warning(row_value(row, "why_risky", "N/A"))
                st.markdown("**Why it may be better or worse than peers**")
                st.write(row_value(row, "why_better_or_worse", "N/A"))
                st.markdown("**Check before pitching**")
                st.write(row_value(row, "what_to_check", "N/A"))

            st.markdown("**Bottom line**")
            st.success(row_value(row, "bottom_line", "N/A"))
            st.caption(
                f"Confidence: {fmt_text(row_value(row, 'confidence_level', 'N/A'))} | "
                f"Evidence quality: {fmt_text(row_value(row, 'evidence_quality', 'N/A'))} | "
                f"Reviewed: {fmt_text(row_value(row, 'evidence_date', 'N/A'))}"
            )

            st.markdown("**Why the score looks like this**")
            render_hebrew_note(
                "למה הציון יצא ככה",
                "הרשימה הבאה מסבירה את הציון במילים פשוטות. למשל: קשר חזק לדאטה סנטרים מעלה ציון, סיכון גבוה או מחיר חם מדי מורידים ציון.",
            )
            for reason in simple_score_explanation(row):
                st.markdown(f"- {reason}")

            with st.expander("More analyst-style pitch text"):
                st.markdown("**One-line pitch**")
                st.write(row_value(row, "pitch_summary", "N/A"))
                st.markdown("**Why it could win**")
                st.write(row_value(row, "why_it_could_win", "N/A"))
                st.markdown("**Main red flags**")
                st.warning(row_value(row, "main_red_flags", "N/A"))
                st.markdown("**What to verify before pitching**")
                st.write(row_value(row, "what_to_verify_next", "N/A"))
                st.markdown("**Old pitch verdict**")
                st.write(row_value(row, "pitch_verdict", "N/A"))

        with tab2:
            render_hebrew_note(
                "Evidence",
                "כאן נמצאות העובדות שהמערכת ניסתה לאמת ממקורות ציבוריים. אם לקוח או חוזה לא מופיעים בדוחות או בפרסומי החברה, המערכת לא ממציאה אותם.",
            )
            st.info(
                "These facts come from public annual reports, SEC filings, company presentations, or official company releases where available. "
                "If a company does not name a customer, the app does not guess."
            )
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**What they own or run**")
                st.write(row_value(row, "portfolio_facts", "Not available in checked public sources."))
                st.markdown("**Lease and income proof**")
                st.write(row_value(row, "leasing_facts", "Not available in checked public sources."))
                st.markdown("**Named customer proof**")
                st.write(row_value(row, "customer_facts", "Not available in checked public sources."))
            with right:
                st.markdown("**Power or size**")
                st.write(row_value(row, "capacity_power_facts", "Not available in checked public sources."))
                st.markdown("**What can grow next**")
                st.write(row_value(row, "expansion_facts", "Not available in checked public sources."))
                st.markdown("**How strong is the proof?**")
                st.write(
                    f"{fmt_text(row_value(row, 'data_quality', 'N/A'))} | "
                    f"checked {fmt_text(row_value(row, 'verified_as_of', 'N/A'))}"
                )
                verified_links = [
                    link.strip()
                    for link in str(row_value(row, "verified_source_links", "")).split(";")
                    if link.strip()
                ]
                if verified_links:
                    st.markdown("**Public sources used**")
                    for link in verified_links:
                        st.markdown(f"- [{link}]({link})")
                st.markdown("**Quiver alternative data**")
                st.write(row_value(row, "alternative_data_signal", "N/A"))
                st.caption("Connected" if is_quiver_connected() else "Quiver not connected")
                if is_quiver_connected():
                    detail = load_quiver_detail(str(row["ticker"]))
                    quiver_rows = []
                    for key, label in [
                        ("congress", "Congress trading"),
                        ("insider", "Insider trading"),
                        ("lobbying", "Lobbying"),
                        ("government_contracts", "Government contracts"),
                        ("news", "Quiver news"),
                    ]:
                        signal = detail.get(key, {})
                        quiver_rows.append(
                            {
                                "Dataset": label,
                                "Status": signal.get("status", "N/A"),
                                "Rows": signal.get("rows", 0),
                            }
                        )
                    st.dataframe(pd.DataFrame(quiver_rows), hide_index=True, use_container_width=True)

            with st.expander("Older notes from the starter CSV"):
                st.markdown("**Original watchlist notes**")
                st.write(row_value(row, "notes", "N/A"))
                st.markdown("**Known customers from CSV**")
                st.write(row_value(row, "known_customers", "N/A"))
                st.markdown("**AI exposure from CSV**")
                st.write(row_value(row, "ai_exposure_notes", "N/A"))

        with tab3:
            render_hebrew_note(
                "Numbers",
                "כאן נמצאים המספרים הפיננסיים והטכניים. לא חייבים להבין הכל. להתחלה כדאי להסתכל על מחיר, שווי שוק, תשואה ל-6 חודשים, RSI, דיבידנד, חוב וצמיחת הכנסות.",
            )
            metrics = pd.DataFrame(
                [
                    ["Currency", fmt_text(row_value(row, "currency", "N/A"))],
                    ["Sector", fmt_text(row_value(row, "sector", "N/A"))],
                    ["Industry", fmt_text(row_value(row, "industry", "N/A"))],
                    ["Price", fmt_price(row_value(row, "price"))],
                    ["Market cap", fmt_market_cap(row_value(row, "market_cap"))],
                    ["Enterprise value", fmt_money(row_value(row, "enterprise_value"))],
                    ["P/E", fmt_number(row_value(row, "pe"))],
                    ["Price/book", fmt_number(row_value(row, "price_to_book"))],
                    ["Dividend yield", fmt_percent(row_value(row, "dividend_yield"))],
                    ["Beta", fmt_number(row_value(row, "beta"))],
                    ["52-week high", fmt_price(row_value(row, "fifty_two_week_high"))],
                    ["52-week low", fmt_price(row_value(row, "fifty_two_week_low"))],
                    ["Distance from 52-week high", fmt_percent(row_value(row, "distance_from_52w_high"))],
                    ["Distance from 52-week low", fmt_percent(row_value(row, "distance_from_52w_low"))],
                    ["Analyst target price", fmt_price(row_value(row, "analyst_target_price"))],
                    ["Analyst recommendation", fmt_text(row_value(row, "analyst_recommendation", "N/A"))],
                    ["Total revenue", fmt_money(row_value(row, "total_revenue"))],
                    ["Revenue growth", fmt_percent(row_value(row, "revenue_growth"))],
                    ["Profit margin", fmt_percent(row_value(row, "profit_margins"))],
                    ["Debt/equity", fmt_number(row_value(row, "debt_to_equity"))],
                    ["Free cash flow", fmt_money(row_value(row, "free_cashflow"))],
                    ["EBITDA", fmt_money(row_value(row, "ebitda"))],
                    ["1M return", fmt_percent(row_value(row, "return_1m"))],
                    ["3M return", fmt_percent(row_value(row, "return_3m"))],
                    ["6M return", fmt_percent(row_value(row, "return_6m"))],
                    ["12M return", fmt_percent(row_value(row, "return_12m"))],
                    ["RSI", fmt_number(row_value(row, "rsi"))],
                    ["50-day moving average", fmt_price(row_value(row, "ma_50"))],
                    ["200-day moving average", fmt_price(row_value(row, "ma_200"))],
                    ["Distance from 50DMA", fmt_percent(row_value(row, "distance_50dma"))],
                    ["Distance from 200DMA", fmt_percent(row_value(row, "distance_200dma"))],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(metrics, hide_index=True, use_container_width=True)
            st.caption("P/E is imperfect for REITs. FFO, AFFO, debt/EBITDA, occupancy, and WALE are better once added.")

        with tab4:
            render_hebrew_note(
                "Compare",
                "כאן משווים את החברה שבחרת לחברה אחרת מהרשימה. זה טוב לשאלה פשוטה: אם רוצים חשיפה לדאטה סנטרים, למה דווקא החברה הזו ולא מתחרה?",
            )
            render_peer_comparison(df, row)

            verified_links = [
                link.strip()
                for link in str(row_value(row, "verified_source_links", "")).split(";")
                if link.strip()
            ]
            if verified_links:
                st.markdown("**Verified sources**")
                for link in verified_links:
                    st.markdown(f"- [{link}]({link})")

            st.markdown("**Sources for this pitch card**")
            links = [link.strip() for link in str(row_value(row, "evidence_links", "")).split(";") if link.strip()]
            if not links:
                st.write("No pitch-specific links yet.")
            else:
                for link in links:
                    st.markdown(f"- [{link}]({link})")

            st.markdown("**Risks**")
            st.write(row_value(row, "risk_notes", "N/A"))
            st.markdown("**Pitch red flags**")
            st.write(row_value(row, "main_red_flags", "N/A"))
            data_error = row_value(row, "data_error", "")
            if data_error:
                st.warning(f"Market data note: {data_error}")
            links = [link.strip() for link in str(row_value(row, "source_links", "")).split(";") if link.strip()]
            if links:
                st.markdown("**Source links**")
                for link in links:
                    st.markdown(f"- [{link}]({link})")


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-top">
                <div>
                    <div class="hero-title">Data Center Real Estate Watchlist</div>
                    <p class="hero-subtitle">
                        A simple research screen for listed data-center REITs, operators, infrastructure owners,
                        and speculative AI/HPC power-secured names. Research tool only. Not financial advice.
                    </p>
                    <p class="hero-subtitle" style="direction: rtl; text-align: right; margin-top: 0.65rem;">
                        בעברית פשוטה: זו מערכת שעוזרת לסדר מניות שקשורות לדאטה סנטרים ולבינה מלאכותית.
                        היא עוזרת להבין מי נראית יציבה, מי מסוכנת, ומה צריך לבדוק לפני שמדברים על השקעה.
                    </p>
                </div>
                <div class="hero-badge">Updated {datetime.now():%Y-%m-%d %H:%M}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()
    render_dad_guide()

    try:
        df = prepare_data()
    except FileNotFoundError:
        st.error("watchlist.csv was not found.")
        return

    filtered = apply_filters(df)
    visible = filtered if not filtered.empty else df

    render_data_source_note(df)

    render_final_five()

    render_summary_cards(visible)

    export_cols = st.columns([1, 3])
    with export_cols[0]:
        pdf_bytes = build_pdf_report(df)
        st.download_button(
            "Export PDF report",
            data=pdf_bytes,
            file_name=f"data_center_watchlist_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with export_cols[1]:
        st.markdown(
            '<div class="simple-note">P/E is shown when available, but it is not ideal for REITs. Better metrics are FFO, AFFO, Price/FFO, debt/EBITDA, occupancy, and WALE.</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        render_panel_start("Research table")
        render_hebrew_note(
            "טבלת מחקר",
            "זו הטבלה המלאה. כל שורה היא חברה. כדאי להתחיל מ-Score, Risk, Category, Simple verdict ו-Pitch verdict. שאר המספרים מיועדים לבדיקה עמוקה יותר.",
        )
        st.dataframe(make_display_table(filtered), hide_index=True, use_container_width=True, height=520)

    render_detail(filtered)


if __name__ == "__main__":
    main()
