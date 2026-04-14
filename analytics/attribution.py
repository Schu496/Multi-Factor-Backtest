"""
analytics/attribution.py
-------------------------
This file answers the question: WHERE did our returns come from?

Our strategy uses 5 factors. But did all 5 contribute equally?
Or did one factor drive most of the performance while others dragged?

"Factor attribution" breaks down the total return into the contribution
from each individual factor. This tells us:

  1. Which factors were genuinely useful over this period
  2. Which factors may have been a drag (negative contribution)
  3. Whether the factor weights in our config make sense

How we calculate it:
  At each rebalance date, we know which stocks were selected and why
  (their factor scores). We can ask: "if we had run a PURE value
  portfolio (ignoring all other factors), what would the return have been?"
  The difference between that pure portfolio and our composite portfolio
  is the value factor's contribution.

  We do this for all 5 factors and the results should roughly add up
  to our total strategy return (with some residual from interactions).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_factor_attribution(
    prices:          pd.DataFrame,
    fundamentals:    pd.DataFrame,
    config:          dict,
    rebalance_log:   list,
) -> pd.DataFrame:
    """
    Estimates the return contribution from each individual factor.

    For each rebalance period, we build a "pure" single-factor portfolio
    using only that factor's scores, compute its return, and record it.
    The average of these returns approximates each factor's contribution.

    Parameters
    ----------
    prices        : full price DataFrame from ingest.py
    fundamentals  : fundamentals DataFrame from ingest.py
    config        : strategy config dictionary
    rebalance_log : list of RebalanceRecord objects from engine.py

    Returns
    -------
    pd.DataFrame
        Columns = factor names, rows = rebalance periods.
        Values = return of a pure single-factor portfolio that period.
    """

    logger.info("Computing factor attribution...")

    factor_names = ["value", "momentum", "quality", "low_volatility", "size"]
    score_cols   = {
        "value":          "value_score",
        "momentum":       "momentum_score",
        "quality":        "quality_score",
        "low_volatility": "low_vol_score",
        "size":           "size_score",
    }

    portfolio_size = config.get("portfolio_size", 50)
    attribution    = {}

    for i, record in enumerate(rebalance_log[:-1]):   # skip last (no forward return)

        rebal_date  = record.date
        next_date   = rebalance_log[i+1].date

        # Get prices for this holding period
        period_prices = prices.loc[rebal_date:next_date]
        if len(period_prices) < 2:
            continue

        # Compute the actual period return for each stock
        period_returns = (
            period_prices.iloc[-1] / period_prices.iloc[0] - 1
        )

        period_attr = {}

        # For each factor, build a pure single-factor portfolio
        for factor_name in factor_names:

            if not config.get("factors", {}).get(factor_name, {}).get("enabled", True):
                continue

            try:
                # Get scores for this factor only
                factor_scores = _get_single_factor_scores(
                    factor_name, prices.loc[:rebal_date], fundamentals, config
                )

                if factor_scores is None or factor_scores.empty:
                    continue

                # Select top N stocks by this factor alone
                top_n   = factor_scores.dropna().nlargest(portfolio_size)
                n       = len(top_n)
                if n == 0:
                    continue

                # Equal-weight this single-factor portfolio
                weight  = 1.0 / n
                tickers = top_n.index.tolist()

                # Compute period return for this pure portfolio
                available = [t for t in tickers if t in period_returns.index]
                if not available:
                    continue

                pure_return = period_returns[available].mean()   # equal weight
                period_attr[factor_name] = pure_return

            except Exception as e:
                logger.debug(f"    Attribution failed for {factor_name} at {rebal_date}: {e}")
                continue

        if period_attr:
            attribution[rebal_date] = period_attr

    if not attribution:
        logger.warning("  No attribution data computed.")
        return pd.DataFrame()

    attr_df = pd.DataFrame(attribution).T
    attr_df.index.name = "date"

    # Summary statistics
    logger.info("\n  Factor Attribution Summary (annualized):")
    logger.info(f"  {'Factor':<20} {'Ann. Return':>12} {'Win Rate':>10} {'Periods':>8}")
    logger.info(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*8}")

    for factor in factor_names:
        if factor not in attr_df.columns:
            continue
        col      = attr_df[factor].dropna()
        if len(col) == 0:
            continue
        ann_ret  = (1 + col.mean()) ** 12 - 1   # monthly → annual
        win_rate = (col > 0).mean()
        logger.info(
            f"  {factor:<20} {ann_ret:>12.2%} {win_rate:>10.1%} {len(col):>8}"
        )

    return attr_df


def _get_single_factor_scores(
    factor_name:   str,
    prices:        pd.DataFrame,
    fundamentals:  pd.DataFrame,
    config:        dict,
) -> pd.Series:
    """
    Helper — computes scores for a single factor in isolation.
    Used internally by compute_factor_attribution.
    """

    try:
        if factor_name == "value":
            from factors.value import compute_value_scores
            return compute_value_scores(fundamentals)

        elif factor_name == "momentum":
            from factors.momentum import compute_momentum_scores
            return compute_momentum_scores(prices)

        elif factor_name == "quality":
            from factors.quality import compute_quality_scores
            return compute_quality_scores(fundamentals)

        elif factor_name == "low_volatility":
            from factors.low_vol import compute_low_vol_scores
            return compute_low_vol_scores(
                prices,
                benchmark_ticker=config.get("benchmark", "SPY")
            )

        elif factor_name == "size":
            from factors.size import compute_size_scores
            return compute_size_scores(fundamentals)

    except Exception as e:
        logger.debug(f"  Single factor score failed for {factor_name}: {e}")
        return None


def compute_attribution_summary(attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes factor attribution into a clean table for reporting.

    Returns a DataFrame with one row per factor showing:
      - Annualized return
      - Hit rate (% of months positive)
      - Best month
      - Worst month
      - Contribution vs equal-weight blend
    """

    if attr_df.empty:
        return pd.DataFrame()

    rows = []
    for factor in attr_df.columns:
        col = attr_df[factor].dropna()
        if len(col) == 0:
            continue

        rows.append({
            "Factor":          factor.replace("_", " ").title(),
            "Ann. Return":     (1 + col.mean()) ** 12 - 1,
            "Hit Rate":        (col > 0).mean(),
            "Best Month":      col.max(),
            "Worst Month":     col.min(),
            "Avg Monthly":     col.mean(),
            "Std Monthly":     col.std(),
        })

    summary = pd.DataFrame(rows).set_index("Factor")
    return summary
