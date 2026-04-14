"""
factors/low_vol.py
------------------
The LOW VOLATILITY factor selects stocks that are unusually calm and
stable — stocks that don't swing wildly up and down.

This seems counterintuitive. In traditional finance theory, more risk
should mean more reward. But decades of real-world data show the opposite:
low-volatility stocks have historically delivered BETTER risk-adjusted
returns than high-volatility ones. This is called the "low volatility anomaly."

Why does it exist? A few explanations:
  - Institutional investors chase exciting, high-risk stocks (lottery effect)
  - This drives up the price of risky stocks and lowers their future returns
  - Boring, stable stocks get ignored and stay attractively priced

We measure low-volatility using two metrics:

  1. Realized Volatility (252-day)
     The standard deviation of daily returns over the past year.
     Annualized by multiplying by √252 (252 trading days in a year).
     LOWER volatility = calmer stock = BETTER score.

  2. Beta vs S&P 500
     How much does the stock move relative to the overall market?
     Beta of 1.0 = moves exactly with the market.
     Beta of 0.5 = moves half as much as the market (defensive).
     Beta of 1.5 = moves 50% more than the market (aggressive).
     LOWER beta = more defensive = BETTER score.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Number of trading days in a year — used to annualize volatility
TRADING_DAYS_PER_YEAR = 252


def compute_low_vol_scores(
    prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
) -> pd.Series:
    """
    Computes low-volatility scores for all stocks.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted closing prices. Rows = dates, columns = tickers.
    benchmark_ticker : str
        The ticker used as the market benchmark for beta calculation.
        Default is "SPY" (S&P 500 ETF).

    Returns
    -------
    pd.Series
        One score per ticker. Higher score = lower volatility = more defensive.
        Index is the ticker symbol.
    """

    logger.info("Computing low-volatility factor scores...")

    # ── Step 1: Compute daily returns ─────────────────────────────────────
    # A "return" is today's price divided by yesterday's price, minus 1.
    # Example: price goes from $100 to $102 → return = (102/100) - 1 = 2%
    # pct_change() does exactly this for every stock at once.
    returns = prices.pct_change().dropna(how="all")

    # ── Step 2: Use only the last 252 trading days (1 year) ───────────────
    # We want recent volatility, not historical. One year is the standard.
    returns_1y = returns.tail(TRADING_DAYS_PER_YEAR)

    if len(returns_1y) < 60:
        logger.warning("  Less than 60 days of return data — vol scores will be limited.")

    # ── Step 3: Realized volatility ───────────────────────────────────────
    # std() computes the standard deviation — a measure of how spread out
    # the daily returns are. Multiply by √252 to annualize it.
    # A stock with 15% annual vol moves roughly ±15% in a typical year.
    daily_std = returns_1y.std()
    annual_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Step 4: Beta calculation ──────────────────────────────────────────
    # Beta measures how much each stock moves relative to the market (SPY).
    # Formula: Beta = Covariance(stock, market) / Variance(market)
    #
    # We loop through each stock and compute this individually.

    betas = {}

    if benchmark_ticker in returns_1y.columns:
        market_returns = returns_1y[benchmark_ticker]
        market_var     = market_returns.var()

        if market_var > 0:
            for ticker in returns_1y.columns:
                stock_returns = returns_1y[ticker].dropna()
                # Align dates — both series must cover the same dates
                aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
                if len(aligned) < 30:
                    betas[ticker] = np.nan
                    continue
                covariance   = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                betas[ticker] = covariance / market_var
        else:
            logger.warning("  Market variance is zero — skipping beta calculation.")
    else:
        logger.warning(f"  Benchmark {benchmark_ticker} not found in price data — skipping beta.")

    beta_series = pd.Series(betas)

    # ── Step 5: Cap beta outliers ─────────────────────────────────────────
    # Very extreme betas (e.g. 5.0 or -2.0) are usually data issues.
    beta_series = beta_series.clip(
        beta_series.quantile(0.01),
        beta_series.quantile(0.99)
    )

    # ── Step 6: Flip signs and rank ───────────────────────────────────────
    # Low volatility = good, but we need high score = good.
    # So we negate: -annual_vol means low-vol stocks get higher (less negative) values.
    vol_score  = (-annual_vol).rank(pct=True)
    beta_score = (-beta_series).rank(pct=True)

    # ── Step 7: Weighted combination ──────────────────────────────────────
    # Weights from strategy.yaml: realized vol = 60%, beta = 40%
    scores = pd.DataFrame({
        "vol_score":  vol_score,
        "beta_score": beta_score
    })

    def weighted_avg(row):
        vals    = [row["vol_score"], row["beta_score"]]
        weights = [0.60,             0.40             ]
        pairs = [(v, w) for v, w in zip(vals, weights) if pd.notna(v)]
        if not pairs:
            return np.nan
        v_arr, w_arr = zip(*pairs)
        return np.average(v_arr, weights=w_arr)

    composite = scores.apply(weighted_avg, axis=1)
    composite.name = "low_vol_score"

    valid = composite.notna().sum()
    logger.info(f"  Low-vol scores computed for {valid} / {len(composite)} tickers.")
    logger.info(f"  Median annualized vol: {annual_vol.median():.1%}")

    return composite
