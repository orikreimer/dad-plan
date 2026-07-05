from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import altair as alt
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
        --bg: #0b1020;
        --panel: #151b31;
        --panel-2: #10172a;
        --line: rgba(148, 163, 184, 0.18);
        --text: #f8fafc;
        --muted: #a9b4c7;
        --blue: #56c7ff;
        --violet: #a76cff;
        --green: #7ee3b0;
        --amber: #f4c76b;
        --red: #ff7d9a;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(86, 199, 255, 0.15), transparent 32rem),
            radial-gradient(circle at 75% 8%, rgba(167, 108, 255, 0.13), transparent 28rem),
            var(--bg);
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
        background: linear-gradient(135deg, rgba(21, 27, 49, 0.96), rgba(13, 19, 35, 0.96));
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28);
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
        color: #dbeafe;
        background: rgba(86, 199, 255, 0.14);
        border: 1px solid rgba(86, 199, 255, 0.35);
        border-radius: 999px;
        padding: 0.45rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .dashboard-card {
        min-height: 126px;
        border-radius: 8px;
        border: 1px solid var(--line);
        background:
            linear-gradient(145deg, rgba(25, 32, 56, 0.98), rgba(13, 19, 35, 0.98)),
            linear-gradient(90deg, rgba(86, 199, 255, 0.28), rgba(167, 108, 255, 0.18));
        padding: 0.95rem;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.22);
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
        color: #cbd5e1;
        font-size: 0.86rem;
        line-height: 1.3;
    }

    .sparkline {
        height: 6px;
        border-radius: 999px;
        margin-top: 0.85rem;
        background: linear-gradient(90deg, var(--blue), var(--violet), var(--green));
        opacity: 0.88;
    }

    .panel {
        border: 1px solid var(--line);
        background: rgba(21, 27, 49, 0.92);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.18);
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
        color: #e5edf8;
        background: rgba(86, 199, 255, 0.1);
        border: 1px solid rgba(86, 199, 255, 0.22);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.65;
        font-size: 0.96rem;
    }

    .hebrew-note strong {
        color: #ffffff;
    }

    .hebrew-note ul {
        margin: 0.45rem 1.2rem 0 0;
        padding: 0;
    }

    .hebrew-note li {
        margin: 0.2rem 0;
    }

    .warning-line {
        color: #ffe4b5;
        background: rgba(244, 199, 107, 0.12);
        border: 1px solid rgba(244, 199, 107, 0.22);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-top: 0.55rem;
        font-size: 0.9rem;
    }

    .good-line {
        color: #d7ffe8;
        background: rgba(126, 227, 176, 0.1);
        border: 1px solid rgba(126, 227, 176, 0.22);
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
        background: rgba(21, 27, 49, 0.98);
        border-color: rgba(148, 163, 184, 0.26);
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
        background: rgba(21, 27, 49, 0.86);
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.16);
    }

    .stDownloadButton button, .stButton button {
        background: linear-gradient(90deg, #2a8fce, #8058d8);
        color: white;
        border: 0;
        border-radius: 8px;
        font-weight: 750;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(21, 27, 49, 0.85);
        border-radius: 8px;
        color: var(--muted);
        padding: 0.45rem 0.8rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--text);
        background: rgba(86, 199, 255, 0.14);
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
    accent_map = {
        "blue": "var(--blue)",
        "violet": "var(--violet)",
        "green": "var(--green)",
        "amber": "var(--amber)",
        "red": "var(--red)",
    }
    color = accent_map.get(accent, "var(--blue)")
    st.markdown(
        f"""
        <div class="dashboard-card" style="border-top: 3px solid {color};">
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


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
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
            compare_groups = adv3.checkbox("Compare stable vs speculative names", value=True)

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

    return filtered, compare_groups


def render_summary_cards(df: pd.DataFrame) -> None:
    ranked = df.sort_values("final_score", ascending=False, na_position="last")
    stable = ranked[ranked["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED])]
    stable_reit = ranked[ranked["category"] == CATEGORY_PURE_PLAY]
    volatile = ranked[ranked["category"] == CATEGORY_VOLATILE]
    income_style = ranked[
        (ranked["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED]))
        & (ranked["dividend_yield"].fillna(0) > 0)
    ]
    overbought_speculative = volatile[volatile["rsi"].apply(_is_number)].sort_values("rsi", ascending=False)
    positive_pe = ranked[ranked["pe"].apply(_is_number) & (ranked["pe"] > 0)]

    cards = [
        (
            "Best overall",
            best_row(ranked),
            lambda row: f"Score {fmt_number(row.get('final_score'))} | {short_category(row.get('category', ''))}",
            "blue",
        ),
        (
            "Best stable choice",
            best_row(stable),
            lambda row: f"Score {fmt_number(row.get('final_score'))} | Risk {row.get('risk_level', 'N/A')}",
            "green",
        ),
        (
            "Best AI/HPC upside",
            best_row(volatile),
            lambda row: f"Speculative | Score {fmt_number(row.get('final_score'))}",
            "violet",
        ),
        (
            "Highest AI upside",
            best_row(ranked, "ai_demand_score"),
            lambda row: f"AI score {fmt_number(row.get('ai_demand_score'))} | {short_category(row.get('category', ''))}",
            "violet",
        ),
        (
            "Cheapest valuation",
            best_row(positive_pe, "pe", ascending=True),
            lambda row: f"P/E {fmt_number(row.get('pe'))} | Yield {fmt_percent(row.get('dividend_yield'))}",
            "amber",
        ),
        (
            "Strongest momentum",
            best_row(ranked, "return_6m"),
            lambda row: f"6M return {fmt_percent(row.get('return_6m'))} | RSI {fmt_number(row.get('rsi'))}",
            "blue",
        ),
        (
            "Most overbought",
            best_row(ranked, "rsi"),
            lambda row: f"RSI {fmt_number(row.get('rsi'))} | Check before chasing",
            "red",
        ),
        (
            "Hot speculative name",
            best_row(overbought_speculative, "rsi"),
            lambda row: f"RSI {fmt_number(row.get('rsi'))} | High risk",
            "red",
        ),
        (
            "Safest pure-play",
            best_row(stable_reit, "safety_score"),
            lambda row: f"Safety {fmt_number(row.get('safety_score'))} | Purity {fmt_number(row.get('data_center_purity_score'))}",
            "green",
        ),
        (
            "Income-style name",
            best_row(income_style, "safety_score"),
            lambda row: f"Yield {fmt_percent(row.get('dividend_yield'))} | Risk {row.get('risk_level', 'N/A')}",
            "amber",
        ),
        (
            "Riskiest AI/HPC",
            best_row(volatile, "safety_score", ascending=True),
            lambda row: f"Safety {fmt_number(row.get('safety_score'))} | {row.get('risk_level', 'N/A')}",
            "red",
        ),
    ]

    for start in range(0, len(cards), 4):
        columns = st.columns(4)
        for col, (title, row, subtitle_fn, accent) in zip(columns, cards[start : start + 4]):
            with col:
                if row is None:
                    render_card(title, "N/A", "", accent)
                else:
                    render_card(title, display_name(row), subtitle_fn(row), accent)

    render_hebrew_note(
        "כרטיסי סיכום",
        "הכרטיסים למעלה הם קיצורי דרך. הם לא המלצה לקנייה. למשל Best stable choice מחפש שם יציב יותר, ו-Hot speculative name מצביע על שם שיכול להיות מעניין אבל מסוכן וחם מדי.",
    )


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(
        strokeOpacity=0
    ).configure_axis(
        labelColor="#cbd5e1",
        titleColor="#a9b4c7",
        gridColor="rgba(148, 163, 184, 0.14)",
        domainColor="rgba(148, 163, 184, 0.2)",
    ).configure_legend(
        labelColor="#cbd5e1",
        titleColor="#a9b4c7",
        orient="bottom",
    ).configure_title(
        color="#f8fafc",
    ).properties(
        background="transparent",
    )


def render_dashboard_panels(df: pd.DataFrame) -> None:
    ranked = df.sort_values("final_score", ascending=False, na_position="last").copy()
    ranked["Short category"] = ranked["category"].map(short_category)
    ranked["Name"] = ranked["ticker"] + "  " + ranked["company_name"]
    top = ranked.head(10)

    left, right = st.columns([1.55, 1])
    with left:
        with st.container(border=True):
            render_panel_start("Top ranked companies")
            render_hebrew_note(
                "גרף דירוג",
                "כאן רואים את החברות שקיבלו את הציון הכללי הגבוה ביותר אחרי הפילטרים. הצבע מראה את סוג החברה, והאורך של העמודה הוא הציון.",
            )
            if top.empty:
                st.info("No companies match the current filters.")
            else:
                chart = (
                    alt.Chart(top)
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                    .encode(
                        y=alt.Y("Name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                        x=alt.X("final_score:Q", title="Final score", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color(
                            "Short category:N",
                            scale=alt.Scale(
                                range=["#56c7ff", "#7ee3b0", "#a76cff", "#f4c76b"]
                            ),
                            title="Category",
                        ),
                        tooltip=[
                            alt.Tooltip("company_name:N", title="Company"),
                            alt.Tooltip("ticker:N", title="Ticker"),
                            alt.Tooltip("final_score:Q", title="Score", format=".1f"),
                            alt.Tooltip("risk_level:N", title="Risk"),
                            alt.Tooltip("label:N", title="Label"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_theme(chart), use_container_width=True)

    with right:
        with st.container(border=True):
            render_panel_start("Risk mix")
            render_hebrew_note(
                "תערובת סיכון",
                "העיגול מראה כמה מהרשימה נמצאות בכל רמת סיכון. אם יש הרבה High או Very high, הרשימה הנוכחית יותר ספקולטיבית.",
            )
            if ranked.empty:
                st.info("No risk data.")
            else:
                risk_counts = ranked.groupby("risk_level").size().reset_index(name="count")
                risk_counts["risk_level"] = pd.Categorical(
                    risk_counts["risk_level"],
                    categories=["Low", "Medium", "High", "Very high"],
                    ordered=True,
                )
                risk_counts = risk_counts.sort_values("risk_level")
                donut = (
                    alt.Chart(risk_counts)
                    .mark_arc(innerRadius=72, outerRadius=116)
                    .encode(
                        theta=alt.Theta("count:Q"),
                        color=alt.Color(
                            "risk_level:N",
                            scale=alt.Scale(range=["#7ee3b0", "#56c7ff", "#f4c76b", "#ff7d9a"]),
                            title="Risk",
                        ),
                        tooltip=[
                            alt.Tooltip("risk_level:N", title="Risk"),
                            alt.Tooltip("count:Q", title="Companies"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_theme(donut), use_container_width=True)

                warning_count = int(ranked["label"].astype(str).str.contains("Overbought|exploded", case=False, na=False).sum())
                speculative_count = int(ranked["label"].astype(str).str.contains("High risk / speculative", na=False).sum())
                st.markdown(
                    f"""
                    <div class="warning-line">Warnings to review: <b>{warning_count}</b></div>
                    <div class="warning-line">Speculative names shown: <b>{speculative_count}</b></div>
                    """,
                    unsafe_allow_html=True,
                )


def render_comparison(df: pd.DataFrame) -> None:
    comparison = df.copy()
    comparison["Group"] = np.where(
        comparison["category"] == CATEGORY_VOLATILE,
        "Volatile AI/HPC",
        "Stable / diversified / indirect",
    )
    grouped = (
        comparison.groupby("Group", as_index=False)
        .agg(
            Companies=("ticker", "count"),
            Avg_score=("final_score", "mean"),
            Avg_6M_return=("return_6m", "mean"),
            Avg_RSI=("rsi", "mean"),
            Avg_purity=("data_center_purity_score", "mean"),
        )
        .fillna(0)
    )

    with st.container(border=True):
        render_panel_start("Stable names vs speculative AI/HPC names")
        render_hebrew_note(
            "השוואה בין יציב לספקולטיבי",
            "לא נכון להשוות REIT יציב לחברת AI/HPC ספקולטיבית כאילו הן אותו דבר. הגרף הזה מפריד בין חברות נדלן ותשתיות יציבות יותר לבין חברות עם סיפור צמיחה מסוכן יותר.",
        )
        chart = (
            alt.Chart(grouped)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Group:N", title=None),
                y=alt.Y("Avg_score:Q", title="Average score", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Group:N", scale=alt.Scale(range=["#56c7ff", "#a76cff"]), legend=None),
                tooltip=[
                    alt.Tooltip("Group:N"),
                    alt.Tooltip("Companies:Q"),
                    alt.Tooltip("Avg_score:Q", format=".1f"),
                    alt.Tooltip("Avg_6M_return:Q", format=".1f"),
                    alt.Tooltip("Avg_RSI:Q", format=".1f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)
        st.markdown(
            '<div class="simple-note">Use this as a sanity check: REITs and infrastructure owners should be judged differently from speculative AI/HPC or bitcoin-mining-transition names.</div>',
            unsafe_allow_html=True,
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

    filtered, compare_groups = apply_filters(df)
    visible = filtered if not filtered.empty else df

    render_data_source_note(df)

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

    render_dashboard_panels(visible)

    if compare_groups:
        render_comparison(visible)

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
