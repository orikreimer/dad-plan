"""PDF export for the Streamlit dashboard."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from scoring import CATEGORY_DIVERSIFIED, CATEGORY_PURE_PLAY, CATEGORY_VOLATILE


def _is_number(value) -> bool:
    try:
        return pd.notna(value) and float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _fmt_number(value, decimals: int = 1) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _fmt_percent(value) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.1f}%"


def _fmt_price(value) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{float(value):,.2f}"


def _fmt_market_cap(value) -> str:
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


def _paragraph(text: str, style) -> Paragraph:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def _summary_table(df: pd.DataFrame, styles, max_rows: int = 5) -> Table:
    columns = ["company_name", "ticker", "category", "risk_level", "final_score", "label"]
    headers = ["Company", "Ticker", "Category", "Risk", "Score", "Label"]
    data = [headers]
    for _, row in df.head(max_rows).iterrows():
        data.append(
            [
                _paragraph(row.get("company_name", ""), styles["Small"]),
                row.get("ticker", ""),
                _paragraph(row.get("category", ""), styles["Small"]),
                row.get("risk_level", ""),
                _fmt_number(row.get("final_score")),
                _paragraph(row.get("label", ""), styles["Small"]),
            ]
        )
    table = Table(data, repeatRows=1, colWidths=[1.7 * inch, 0.75 * inch, 2.2 * inch, 0.75 * inch, 0.65 * inch, 2.0 * inch])
    table.setStyle(_table_style())
    return table


def _full_watchlist_table(df: pd.DataFrame, styles) -> Table:
    headers = ["Company", "Ticker", "Risk", "Price", "Mkt Cap", "6M", "RSI", "Yield", "P/E", "Score", "Label"]
    data = [headers]
    for _, row in df.iterrows():
        data.append(
            [
                _paragraph(row.get("company_name", ""), styles["Tiny"]),
                row.get("ticker", ""),
                row.get("risk_level", ""),
                _fmt_price(row.get("price")),
                _fmt_market_cap(row.get("market_cap")),
                _fmt_percent(row.get("return_6m")),
                _fmt_number(row.get("rsi")),
                _fmt_percent(row.get("dividend_yield")),
                _fmt_number(row.get("pe")),
                _fmt_number(row.get("final_score")),
                _paragraph(row.get("label", ""), styles["Tiny"]),
            ]
        )
    table = Table(
        data,
        repeatRows=1,
        colWidths=[1.55 * inch, 0.65 * inch, 0.7 * inch, 0.65 * inch, 0.75 * inch, 0.55 * inch, 0.45 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 1.8 * inch],
    )
    table.setStyle(_table_style(font_size=6.2, leading=7.4))
    return table


def _pitch_notes_table(df: pd.DataFrame, styles, max_rows: int = 5) -> Table:
    headers = ["Company", "Ticker", "Bottom Line", "Simple Answer", "Check Before Pitching"]
    data = [headers]
    for _, row in df.head(max_rows).iterrows():
        data.append(
            [
                _paragraph(row.get("company_name", ""), styles["Tiny"]),
                row.get("ticker", ""),
                _paragraph(row.get("bottom_line", row.get("pitch_verdict", "N/A")), styles["Tiny"]),
                _paragraph(row.get("simple_answer", row.get("pitch_summary", "N/A")), styles["Tiny"]),
                _paragraph(row.get("what_to_check", row.get("what_to_verify_next", "N/A")), styles["Tiny"]),
            ]
        )
    table = Table(
        data,
        repeatRows=1,
        colWidths=[1.45 * inch, 0.55 * inch, 1.55 * inch, 3.15 * inch, 3.35 * inch],
    )
    table.setStyle(_table_style(font_size=6.2, leading=7.4))
    return table


def _verified_facts_table(df: pd.DataFrame, styles, max_rows: int = 8) -> Table:
    headers = ["Company", "Ticker", "Portfolio / Leasing", "Customers", "Power / Expansion"]
    data = [headers]
    for _, row in df.head(max_rows).iterrows():
        portfolio = f"{row.get('portfolio_facts', 'N/A')} {row.get('leasing_facts', 'N/A')}"
        power = f"{row.get('capacity_power_facts', 'N/A')} {row.get('expansion_facts', 'N/A')}"
        data.append(
            [
                _paragraph(row.get("company_name", ""), styles["Tiny"]),
                row.get("ticker", ""),
                _paragraph(portfolio, styles["Tiny"]),
                _paragraph(row.get("customer_facts", "N/A"), styles["Tiny"]),
                _paragraph(power, styles["Tiny"]),
            ]
        )
    table = Table(
        data,
        repeatRows=1,
        colWidths=[1.15 * inch, 0.55 * inch, 3.1 * inch, 2.35 * inch, 3.0 * inch],
    )
    table.setStyle(_table_style(font_size=5.8, leading=6.8))
    return table


def _table_style(font_size: float = 7.5, leading: float = 8.5) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), leading),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def build_pdf_report(df: pd.DataFrame, generated_at: datetime | None = None) -> bytes:
    """Build a PDF report and return it as bytes."""
    generated_at = generated_at or datetime.now()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title="Data Center Real Estate Watchlist",
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "Title",
            parent=base_styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=10,
        ),
        "Heading": ParagraphStyle(
            "Heading",
            parent=base_styles["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle("Body", parent=base_styles["BodyText"], fontSize=8.5, leading=11),
        "Small": ParagraphStyle("Small", parent=base_styles["BodyText"], fontSize=7, leading=8),
        "Tiny": ParagraphStyle("Tiny", parent=base_styles["BodyText"], fontSize=6.2, leading=7.4),
    }

    story = []
    story.append(Paragraph("Data Center Real Estate Watchlist", styles["Title"]))
    story.append(_paragraph(f"Generated: {generated_at:%Y-%m-%d %H:%M}", styles["Body"]))
    story.append(_paragraph("Research tool only. Not financial advice.", styles["Body"]))
    story.append(Spacer(1, 0.1 * inch))

    ranked = df.sort_values("final_score", ascending=False, na_position="last").copy()
    story.append(Paragraph("Top 5 Ranked Companies Overall", styles["Heading"]))
    story.append(_summary_table(ranked, styles, max_rows=5))

    stable = ranked[ranked["category"].isin([CATEGORY_PURE_PLAY, CATEGORY_DIVERSIFIED])]
    story.append(Paragraph("Best Stable / Real-Estate-Style Data-Center Companies", styles["Heading"]))
    story.append(_summary_table(stable, styles, max_rows=5) if not stable.empty else _paragraph("N/A", styles["Body"]))

    volatile = ranked[ranked["category"] == CATEGORY_VOLATILE]
    story.append(Paragraph("Best Volatile / Speculative AI-HPC Infrastructure Companies", styles["Heading"]))
    story.append(_summary_table(volatile, styles, max_rows=5) if not volatile.empty else _paragraph("N/A", styles["Body"]))

    if {"simple_answer", "bottom_line", "what_to_check"}.issubset(ranked.columns) or {"pitch_summary", "pitch_verdict", "what_to_verify_next"}.issubset(ranked.columns):
        story.append(Paragraph("Plain-English Pitch Notes For Top Ranked Companies", styles["Heading"]))
        story.append(_pitch_notes_table(ranked, styles, max_rows=5))

    if {"portfolio_facts", "leasing_facts", "customer_facts", "capacity_power_facts", "expansion_facts"}.issubset(ranked.columns):
        story.append(Paragraph("Verified Company Facts For Top Ranked Companies", styles["Heading"]))
        story.append(_verified_facts_table(ranked, styles, max_rows=8))

    warnings = []
    if "label" in ranked.columns:
        overbought = int(ranked["label"].astype(str).str.contains("Overbought warning", na=False).sum())
        exploded = int(ranked["label"].astype(str).str.contains("May have already exploded", na=False).sum())
        missing = int(ranked["label"].astype(str).str.contains("Data missing", na=False).sum())
        speculative = int(ranked["label"].astype(str).str.contains("High risk / speculative", na=False).sum())
        warnings = [
            f"Overbought warnings: {overbought}",
            f"May have already exploded: {exploded}",
            f"Data missing: {missing}",
            f"High risk / speculative labels: {speculative}",
        ]

    story.append(Paragraph("Key Warnings", styles["Heading"]))
    story.append(_paragraph("; ".join(warnings) if warnings else "No warning summary available.", styles["Body"]))

    story.append(Paragraph("Scoring Model", styles["Heading"]))
    story.append(
        _paragraph(
            "Score is 0-100 using valuation 25%, momentum 20%, data-center purity 20%, AI demand exposure 20%, and safety/risk 15%. "
            "The model penalizes missing data, RSI above 75, six-month returns above 150%, very high risk, vague indirect exposure, and crypto-heavy models.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Category Notes", styles["Heading"]))
    story.append(
        _paragraph(
            "Pure-play REIT/operators are not the same type of investment as volatile AI/HPC or crypto-transition infrastructure names. "
            "The speculative group may have power-secured sites or high-density campus upside, but the risk profile can be much higher than REITs.",
            styles["Body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Full Watchlist Table", styles["Heading"]))
    story.append(_full_watchlist_table(ranked, styles))

    doc.build(story)
    return buffer.getvalue()
