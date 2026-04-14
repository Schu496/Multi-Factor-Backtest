# Multi-Factor Equity Strategy Backtest

A professional-grade, modular Python framework for constructing, backtesting, and evaluating multi-factor equity investment strategies — built to institutional standards.

**Backtest period:** January 2010 – December 2024 (15 years) | **Universe:** S&P 500 | **Benchmark:** SPY

---

## Strategy Results

| Metric | Strategy | Benchmark (SPY) |
|---|---|---|
| **CAGR** | **11.75%** | 12.65% |
| **Total Return** | **383.63%** | 441.50% |
| **Annualized Volatility** | **15.11%** | 17.01% |
| **Sharpe Ratio** | **0.61** | 0.61 |
| **Sortino Ratio** | **0.74** | — |
| **Max Drawdown** | **-38.32%** | -36.47% |
| **Alpha (annual)** | **+1.14%** | — |
| **Beta** | **0.66** | 1.00 |
| **Monthly Win Rate** | **66.48%** | — |

> The strategy matched SPY's risk-adjusted returns (Sharpe 0.61 vs 0.61) while running at lower volatility (15.1% vs 17.0%) and generating positive alpha (+1.14% annually), demonstrating that the factor model adds genuine value beyond market exposure.

---

## Tearsheet Preview

*Run `python main.py` to generate the full 4-page PDF tearsheet in `reporting/output/`*

The tearsheet includes:
- **Page 1** — Full performance summary table (strategy vs benchmark)
- **Page 2** — Cumulative returns and drawdown chart (2010–2024)
- **Page 3** — Rolling 12-month Sharpe ratio and annual returns comparison
- **Page 4** — Monthly returns heatmap and factor attribution breakdown

---

## Strategy Overview

This framework implements a **5-factor long-only equity model** applied to the S&P 500 universe. Stocks are ranked monthly on a composite factor score and assembled into a 50-stock equal-weighted portfolio, rebalanced monthly.

| Factor | Weight | Metric(s) | Signal |
|---|---|---|---|
| **Value** | 25% | P/B ratio, EV/EBITDA, P/E ratio | Low = Cheap = Positive |
| **Momentum** | 25% | 12-1 month price return | High = Strong = Positive |
| **Quality** | 20% | ROE, Gross Margin, Debt/Equity | High Quality = Positive |
| **Low Volatility** | 15% | 252-day realized vol, Beta | Low = Defensive = Positive |
| **Size** | 15% | Log Market Cap | Small Cap = Positive |

---

## Project Architecture

```
multi_factor_backtest/
│
├── data/                       # Data layer
│   ├── ingest.py               # Price + fundamental data (yfinance)
│   └── universe.py             # S&P 500 constituent management
│
├── factors/                    # Factor construction
│   ├── value.py                # P/B, EV/EBITDA, P/E
│   ├── momentum.py             # 12-1 month return
│   ├── quality.py              # ROE, margins, leverage
│   ├── low_vol.py              # Realized vol, beta
│   ├── size.py                 # Log market cap
│   └── composite.py            # Z-score normalization + weighting
│
├── portfolio/                  # Portfolio construction
│   ├── construction.py         # Stock selection (top 50)
│   └── weighting.py            # Equal weight / risk parity
│
├── backtest/                   # Backtesting engine
│   ├── engine.py               # Monthly simulation loop
│   └── rebalance.py            # Trade execution + transaction costs
│
├── analytics/                  # Performance measurement
│   ├── performance.py          # CAGR, Sharpe, Sortino, Max Drawdown
│   ├── risk.py                 # VaR, CVaR, rolling metrics
│   └── attribution.py          # Factor return attribution
│
├── reporting/                  # Output generation
│   ├── tearsheet.py            # 4-page PDF tearsheet
│   └── charts.py               # Matplotlib visualizations
│
├── tests/
│   └── test_pipeline.py        # Smoke test (10 tickers, 2 years)
│
├── configs/
│   └── strategy.yaml           # All tunable parameters
│
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/multi-factor-backtest.git
cd multi-factor-backtest

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the smoke test first (10 tickers, ~3 minutes)
python tests/test_pipeline.py

# 5. Run the full 15-year backtest (~45 minutes first run, cached after)
python main.py
```

> **First run:** Downloads ~15 years of price and fundamental data for 500 stocks (~40 min). All subsequent runs load from cache in seconds.

---

## Configuration

All strategy parameters are controlled via `configs/strategy.yaml`:

```yaml
universe:             sp500
start_date:           "2010-01-01"
end_date:             "2024-12-31"
benchmark:            "SPY"
portfolio_size:       50
weighting:            equal_weight     # or: risk_parity
rebalance_freq:       monthly          # or: quarterly
transaction_cost_bps: 10

factors:
  value:          { enabled: true, weight: 0.25 }
  momentum:       { enabled: true, weight: 0.25 }
  quality:        { enabled: true, weight: 0.20 }
  low_volatility: { enabled: true, weight: 0.15 }
  size:           { enabled: true, weight: 0.15 }
```

---

## Factor Attribution (2010–2024)

| Factor | Ann. Return | Win Rate | Periods |
|---|---|---|---|
| Quality | +26.2% | 68.7% | 179 |
| Momentum | +24.6% | 67.7% | 167 |
| Size | +15.6% | 66.5% | 179 |
| Value | +13.3% | 62.0% | 179 |
| Low Volatility | +13.3% | 67.6% | 179 |

Quality and Momentum were the strongest contributors over the full period, consistent with the academic literature on factor investing.

---

## Tech Stack

| Module | Library |
|---|---|
| Data ingestion | `yfinance`, `requests`, `beautifulsoup4` |
| Data processing | `pandas`, `numpy` |
| Factor construction | `pandas`, `numpy`, `scipy` |
| Portfolio construction | `numpy`, `scipy.optimize` |
| Backtesting | Custom engine |
| Performance analytics | `quantstats`, custom |
| Visualization | `matplotlib` |
| Config management | `pyyaml` |
| Testing | `pytest` |

---

## Investment Universe

- **Universe:** S&P 500 constituents
- **Backtest period:** January 2010 – December 2024 (15 years)
- **Rebalancing:** Monthly (last trading day of each month)
- **Benchmark:** SPY (S&P 500 ETF)
- **Portfolio size:** Top 50 stocks by composite factor score
- **Weighting:** Equal-weighted
- **Transaction costs:** 10 bps per trade (one-way)

---

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Nothing in this repository constitutes investment advice.

---

## License

MIT License — free to use, modify, and distribute with attribution.
