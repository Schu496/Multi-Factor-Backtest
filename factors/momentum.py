"""
factors/momentum.py
-------------------
The MOMENTUM factor captures the tendency of recent winners to keep winning.

We use "12-1 momentum": the return over the past 12 months skipping
the most recent month. Skipping the last month avoids short-term reversal.

HIGHER return = STRONGER momentum = BETTER score.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_momentum_scores(prices: pd.DataFrame) -> pd.Series:
    """
    Computes 12-1 month momentum scores for all stocks.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted closing prices. Rows = dates, columns = tickers.

    Returns
    -------
    pd.Series
        One score per ticker. Higher score = stronger momentum.
    """

    logger.info("Computing momentum factor scores...")

    # Resample to month-end prices to reduce noise
    monthly = prices.resample("ME").last()

    if len(monthly) < 13:
        logger.warning("  Not enough price history for momentum (need 13+ months).")
        return pd.Series(dtype=float)

    # 12-1 momentum: price 1 month ago vs price 13 months ago
    price_1m_ago  = monthly.iloc[-1]    # most recent month-end
    price_13m_ago = monthly.iloc[-13]   # 12 months before that

    # Avoid division by zero or negative base prices
    valid_base     = price_13m_ago.replace(0, np.nan)
    valid_base     = valid_base.where(valid_base > 0)
    momentum_return = (price_1m_ago / valid_base) - 1

    # Winsorize at 1st/99th percentile to remove data errors
    lower = momentum_return.quantile(0.01)
    upper = momentum_return.quantile(0.99)
    momentum_return = momentum_return.clip(lower, upper)

    # Rank to 0-1 scale (higher return = higher rank = better)
    momentum_score      = momentum_return.rank(pct=True)
    momentum_score.name = "momentum_score"

    valid = momentum_score.notna().sum()
    logger.info(f"  Momentum scores computed for {valid} / {len(momentum_score)} tickers.")

    if valid > 0:
        valid_returns = momentum_return.dropna()
        logger.info(f"  Return range: {valid_returns.min():.1%} to {valid_returns.max():.1%}")

    return momentum_score
