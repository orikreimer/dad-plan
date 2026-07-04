# Data Center Real Estate Watchlist

Simple Streamlit MVP for researching listed data-center real estate and digital infrastructure stocks.

The goal is a clean research screen for physical data-center exposure: companies that own, operate, lease, finance, or control data centers and related infrastructure. It is not a trading app and it is not focused on chips, GPUs, servers, or software companies.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional Quiver setup:

```bash
cp .env.example .env
# add QUIVER_AUTH_TOKEN=your_token_here
```

The app works without a Quiver key and will show `Quiver not connected`.

## Files

- `app.py`: Streamlit dashboard, filters, summary cards, detail view, and PDF download.
- `watchlist.csv`: Manual company universe, categories, risk levels, notes, customers, AI exposure, risks, and source links.
- `pitch_research.csv`: Manual pitch layer with thesis, expansion/leasing evidence, customer/demand evidence, red flags, peer-comparison logic, verification checklist, and source links.
- `plain_language_research.csv`: Simple dad-friendly explanation layer for each stock: quick answer, why it can work, why it can fail, future story, what to check, and bottom line.
- `verified_company_data.csv`: Checked company facts for each ticker from public annual reports, SEC filings, official presentations, and official company releases: portfolio size, leasing, customers, power/capacity, expansion, data quality, and source links.
- `data_sources.py`: yfinance market data, returns, RSI, moving averages, and safe missing-data handling.
- `quiver.py`: Placeholder structure for future Quiver Quant integration.
- `scoring.py`: Explainable 0-100 scoring model and labels.
- `report_export.py`: ReportLab PDF generation.
- `.env.example`: API key template.
- `requirements.txt`: Python dependencies.

## Current Data Model

Automated in the MVP:

- Price
- Market cap
- Enterprise value if available
- P/E if available
- Price/book if available
- Dividend yield if available
- Beta if available
- 52-week high / low and distance from each
- Analyst target price and recommendation if available
- Revenue, revenue growth, profit margin, debt/equity, free cash flow, and EBITDA if available
- 1M / 3M / 6M / 12M returns
- RSI
- 50-day moving average
- 200-day moving average
- Basic score

Manual in `watchlist.csv` for now:

- Category
- Risk level
- Data-center purity score
- Known customers
- AI exposure notes
- Risk notes
- Source links
- REIT-specific details like FFO, AFFO, occupancy, WALE, MW capacity

Manual in `pitch_research.csv` for now:

- One-line pitch
- Why it could win
- Expansion or leasing evidence
- Customer or demand evidence
- Power and capacity notes
- Why it may be better than peers
- Main red flags
- What to verify before pitching
- Dad-friendly pitch verdict
- Source links and evidence quality

Manual in `plain_language_research.csv` for now:

- Simple answer
- Why it can be good
- Why it can be risky
- Future story
- Why it may be better or worse than peers
- What to check before pitching
- Bottom line

Checked in `verified_company_data.csv` from public company sources:

- Portfolio facts
- Leasing facts
- Customer facts
- Power / capacity facts
- Expansion facts
- Source links

These are not guessed relationships. If a company does not publicly name a customer, the app says that the customer is not disclosed. If a public website blocks automated access, the row says that clearly instead of filling in fake detail.

If a company does not publicly disclose a tenant, lease, or exact power number, the app should say that clearly. It should not guess customer names.

Customer relationships are intentionally conservative. The CSV uses `unknown`, `not disclosed`, or verification language unless a relationship should be confirmed from public filings or company releases.

## Categories

1. `Pure-play data center REIT/operator`
   Companies mainly focused on owning or operating data centers.

2. `Diversified real estate / infrastructure with data-center exposure`
   Companies with meaningful data-center or digital infrastructure exposure, but not pure data-center businesses.

3. `Volatile AI/HPC infrastructure / power-secured data-center play`
   More speculative companies often connected to bitcoin mining, HPC, AI cloud, power-secured campuses, or high-density sites.

4. `Asset manager / indirect exposure`
   Companies that own, finance, or manage data-center infrastructure but are not pure operating companies.

Volatile AI/HPC names are not the same type of investment as REITs. They can have high upside from power or campus control, but financing, dilution, construction, customer, and crypto-cycle risks can be much higher.

## Scoring

The score is 0-100:

- Valuation: 25%
- Momentum: 20%
- Data-center purity: 20%
- AI demand exposure: 20%
- Safety/risk: 15%

The model rewards high data-center purity, credible AI/cloud/HPC exposure, stable REIT/operator profiles, diversification, and dividends.

It penalizes:

- Missing market data
- RSI above 75
- 6M return above 150%
- Very high risk
- Vague or indirect models
- Crypto-heavy exposure

Labels include:

- `Strong candidate`
- `Interesting`
- `Watchlist only`
- `High risk / speculative`
- `Data missing`
- `Overbought warning`
- `May have already exploded`

## REIT Metric Note

P/E is not ideal for REITs because depreciation can make accounting earnings less useful.

Better REIT metrics include:

- FFO
- AFFO
- Price/FFO
- Price/AFFO
- Dividend yield
- Debt/EBITDA
- Occupancy
- WALE

The MVP shows P/E when yfinance provides it, but it should be treated as imperfect.

## Data Sources

Current prototype:

- `yfinance` for prototype market data from Yahoo Finance's public data interface
- Public annual reports, SEC filings, official investor presentations, and official company releases for company-specific portfolio, leasing, customer, power, and expansion facts
- Manual CSV for classification and qualitative research notes

Important source warning: `yfinance` is useful and cheap for an MVP, but it is not a paid institutional data feed. The app treats missing values as `N/A`. For any real-money decision, verify important facts in company filings, investor presentations, exchange announcements, or a higher-grade market data source.

Possible future sources:

- Quiver Quant API for alternative data
- Financial Modeling Prep for fundamentals and ratios
- Alpha Vantage for price and technical indicators
- Finnhub for fundamentals and market data
- SEC EDGAR API for filings
- Company IR pages and annual reports for REIT-specific metrics like FFO, AFFO, occupancy, WALE, MW capacity, and tenant concentration

## Future TODOs

- Add FMP API for valuation ratios.
- Add Alpha Vantage API as backup.
- Add Finnhub API.
- Add SEC filing parser.
- Add automatic news monitoring for OpenAI, Oracle, Microsoft, Meta, CoreWeave, Stargate, hyperscaler leases, MW capacity, and pre-leasing.
- Add manual customer relationship database.
- Add charts.
- Add database instead of CSV.
- Add scheduled daily refresh.
- Add login later if needed.
- Add better PDF formatting.
- Add FFO/AFFO metrics for REITs.
- Add tenant concentration analysis.
- Add power capacity / MW tracking.

## Limitations

`yfinance` is convenient for prototypes, but it can be incomplete, delayed, inconsistent across international listings, or temporarily unavailable. Missing values are shown as `N/A` and should not crash the app.

This dashboard is a research tool only. It is not financial advice.
