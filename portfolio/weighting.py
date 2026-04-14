"""
portfolio/weighting.py
-----------------------
Once we know WHICH stocks to buy, this file decides HOW MUCH of each
stock to buy — i.e. the portfolio weights.

We implement two approaches:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. EQUAL WEIGHT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The simplest possible approach: divide your money equally.
If you hold 50 stocks, put 2% in each one (1/50 = 2%).

Pros:
  - Dead simple, no estimation needed
  - Naturally tilts toward smaller stocks (buys same dollar amount
    regardless of company size)
  - Surprisingly hard to beat in practice

Cons:
  - Ignores the fact that some stocks are far riskier than others
  - A 2% position in a wildly volatile stock contributes far more
    risk than a 2% position in a stable stock

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2. RISK PARITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each stock gets a weight inversely proportional to its volatility.
A stock with 30% annual volatility gets half the weight of a stock
with 15% annual volatility. This way each stock contributes roughly
the same amount of RISK to the portfolio — hence "risk parity."

Example with 3 stocks:
  Stock A: vol = 10% → raw weight = 1/10 = 0.100
  Stock B: vol = 20% → raw weight = 1/20 = 0.050
  Stock C: vol = 40% → raw weight = 1/40 = 0.025
  Total = 0.175 → normalize to sum to 1.0:
  Stock A: 0.100/0.175 = 57.1%
  Stock B: 0.050/0.175 = 28.6%
  Stock C: 0.025/0.175 = 14.3%

Stock A (calmest) gets the biggest allocation. Stock C (most volatile)
gets the smallest. The result: each stock contributes similar risk.

Pros:
  - Better risk management than equal weight
  - Reduces the impact of any single volatile stock blowing up

Cons:
  - Requires price history to estimate volatility
  - Can over-weight very low-vol stocks and under-weight high-vol ones
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Number of trading days per year — used to annualize volatility
TRADING_DAYS = 252


def equal_weight(tickers: list[str]) -> dict[str, float]:
    """
    Assigns equal weight to every stock in the portfolio.

    Parameters
    ----------
    tickers : list of ticker symbols selected by construction.py

    Returns
    -------
    dict
        {ticker: weight} where all weights are equal and sum to 1.0.

    Example (3 stocks):
        {'AAPL': 0.333, 'MSFT': 0.333, 'GOOGL': 0.333}
    """

    if not tickers:
        logger.error("  Cannot weight an empty ticker list.")
        return {}

    n = len(tickers)
    w = 1.0 / n

    weights = {ticker: w for ticker in tickers}

    logger.info(f"  Equal weight: {n} stocks × {w:.4f} ({w:.2%} each)")

    # Sanity check: weights must sum to 1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0!"

    return weights


def risk_parity_weight(
    tickers:      list[str],
    prices:       pd.DataFrame,
    lookback_days: int = TRADING_DAYS,
    min_weight:   float = 0.001,
    max_weight:   float = 0.10,
) -> dict[str, float]:
    """
    Assigns weights inversely proportional to each stock's volatility.

    Parameters
    ----------
    tickers : list of ticker symbols selected by construction.py
    prices  : daily adjusted closing prices DataFrame from ingest.py
    lookback_days : how many trading days of history to use for vol estimate
                    Default is 252 (one year)
    min_weight : minimum weight for any single stock (floor)
                 Prevents a stock from having a near-zero allocation
    max_weight : maximum weight for any single stock (cap)
                 Prevents one very calm stock from dominating the portfolio

    Returns
    -------
    dict
        {ticker: weight} where weights are inversely proportional to vol
        and sum to 1.0.
    """

    if not tickers:
        logger.error("  Cannot weight an empty ticker list.")
        return {}

    logger.info(f"  Computing risk parity weights for {len(tickers)} stocks...")

    # ── Step 1: Filter prices to only our selected stocks ─────────────────
    # Only keep columns (tickers) that are in our selected portfolio
    available = [t for t in tickers if t in prices.columns]
    missing   = [t for t in tickers if t not in prices.columns]

    if missing:
        logger.warning(f"  {len(missing)} tickers missing from price data: {missing[:5]}")

    if not available:
        logger.warning("  No tickers found in price data — falling back to equal weight.")
        return equal_weight(tickers)

    # ── Step 2: Compute daily returns for the lookback period ─────────────
    recent_prices = prices[available].tail(lookback_days + 1)
    returns       = recent_prices.pct_change().dropna()

    if len(returns) < 20:
        logger.warning(f"  Only {len(returns)} days of returns — falling back to equal weight.")
        return equal_weight(tickers)

    # ── Step 3: Compute annualized volatility for each stock ──────────────
    # std() = standard deviation of daily returns
    # Multiply by √252 to convert from daily to annual
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    # ── Step 4: Handle any stocks with zero or missing volatility ─────────
    # This can happen if a stock had no price movement (rare but possible)
    annual_vol = annual_vol.replace(0, np.nan)
    annual_vol = annual_vol.dropna()

    if len(annual_vol) == 0:
        logger.warning("  All volatilities are zero/NaN — falling back to equal weight.")
        return equal_weight(tickers)

    # ── Step 5: Compute raw inverse-volatility weights ────────────────────
    # Lower vol → larger 1/vol → larger raw weight → larger final weight
    inv_vol     = 1.0 / annual_vol
    raw_weights = inv_vol / inv_vol.sum()   # normalize to sum to 1.0

    # ── Step 6: Apply weight constraints (floor and cap) ──────────────────
    # Without constraints, very low-vol stocks can get 15-20% weights,
    # concentrating the portfolio dangerously.
    raw_weights = raw_weights.clip(lower=min_weight, upper=max_weight)

    # Re-normalize after clipping so weights still sum to 1.0
    raw_weights = raw_weights / raw_weights.sum()

    # ── Step 7: Handle tickers that had no price data ─────────────────────
    # Any tickers in our list but not in prices get equal share of
    # whatever weight is left over (shouldn't happen often)
    weights_dict = raw_weights.to_dict()

    if missing:
        # Give missing tickers the average weight of what we computed
        avg_weight   = 1.0 / len(tickers)
        missing_total = avg_weight * len(missing)
        # Scale down existing weights proportionally
        scale = 1.0 - missing_total
        weights_dict = {k: v * scale for k, v in weights_dict.items()}
        for t in missing:
            weights_dict[t] = avg_weight

    # Final normalization to guarantee sum = 1.0
    total = sum(weights_dict.values())
    weights_dict = {k: v / total for k, v in weights_dict.items()}

    # ── Step 8: Log summary ───────────────────────────────────────────────
    w_series = pd.Series(weights_dict)
    logger.info(f"  Risk parity weights computed:")
    logger.info(f"    Min weight : {w_series.min():.2%} ({w_series.idxmin()})")
    logger.info(f"    Max weight : {w_series.max():.2%} ({w_series.idxmax()})")
    logger.info(f"    Mean weight: {w_series.mean():.2%}")
    logger.info(f"    Sum of weights: {w_series.sum():.6f}")

    # Final sanity check
    assert abs(w_series.sum() - 1.0) < 1e-6, f"Weights sum to {w_series.sum()}, not 1.0!"

    return weights_dict


def get_weights(
    tickers:       list[str],
    weighting:     str,
    prices:        pd.DataFrame = None,
) -> dict[str, float]:
    """
    Dispatcher function — chooses the right weighting method based on
    the 'weighting' setting from strategy.yaml.

    Parameters
    ----------
    tickers   : list of selected tickers from construction.py
    weighting : string from config — 'equal_weight' or 'risk_parity'
    prices    : price DataFrame (only needed for risk_parity)

    Returns
    -------
    dict : {ticker: weight} summing to 1.0
    """

    logger.info(f"  Applying weighting scheme: {weighting}")

    if weighting == "equal_weight":
        return equal_weight(tickers)

    elif weighting == "risk_parity":
        if prices is None:
            logger.warning("  Risk parity requires price data — falling back to equal weight.")
            return equal_weight(tickers)
        return risk_parity_weight(tickers, prices)

    else:
        logger.warning(f"  Unknown weighting '{weighting}' — defaulting to equal weight.")
        return equal_weight(tickers)
