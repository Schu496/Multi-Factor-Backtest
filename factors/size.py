"""
factors/size.py
---------------
The SIZE factor tilts the portfolio toward smaller companies.

The "size premium" is one of the oldest documented effects in finance,
first published by Fama and French in 1992. Smaller companies have
historically delivered higher returns than large companies over the
long run — though with more volatility along the way.

Why do small caps outperform?
  - Less analyst coverage means they are more likely to be mispriced
  - They can grow faster from a smaller base
  - Institutional investors often can't buy them (too small to move the needle),
    so they are systematically undervalued

Important note for our project: we are working within the S&P 500, which
only contains large companies. So we are not truly buying small-caps —
we are tilting toward the SMALLER end of large-caps (e.g. preferring a
$10B company over a $500B company within the index).

Metric: Log Market Capitalization
  Market cap = share price × number of shares outstanding
  We use the NATURAL LOG of market cap because the raw values span an
  enormous range ($10B to $3T). The log compresses this range so that
  the difference between $10B and $20B gets similar weight as the
  difference between $500B and $1T.

LOWER log market cap = smaller company = BETTER score (size premium).
We flip the sign so higher score = smaller = better.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_size_scores(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Computes size factor scores for each stock.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        The fundamentals DataFrame from data/ingest.py.
        Must contain column: market_cap (or log_market_cap).

    Returns
    -------
    pd.Series
        One score per ticker. Higher score = smaller company = more size premium.
        Index is the ticker symbol.
    """

    logger.info("Computing size factor scores...")

    # ── Step 1: Get log market cap ────────────────────────────────────────
    # We already computed log_market_cap in ingest.py, but we recompute
    # it here in case it's missing, as a safety net.

    if "log_market_cap" in fundamentals.columns:
        log_mktcap = fundamentals["log_market_cap"].copy()
    elif "market_cap" in fundamentals.columns:
        # Replace 0s and negatives with NaN before taking log
        mktcap     = fundamentals["market_cap"].replace(0, np.nan)
        mktcap     = mktcap.where(mktcap > 0)
        log_mktcap = np.log(mktcap)
    else:
        logger.error("  Neither 'market_cap' nor 'log_market_cap' found in fundamentals.")
        return pd.Series(dtype=float)

    # ── Step 2: Remove missing values ─────────────────────────────────────
    log_mktcap = log_mktcap.replace([np.inf, -np.inf], np.nan)

    # ── Step 3: Log some summary stats to help with debugging ────────────
    valid_caps = np.exp(log_mktcap.dropna())
    if len(valid_caps) > 0:
        logger.info(f"  Market cap range: ${valid_caps.min()/1e9:.1f}B — ${valid_caps.max()/1e9:.1f}T")
        logger.info(f"  Median market cap: ${valid_caps.median()/1e9:.1f}B")

    # ── Step 4: Flip sign and rank ────────────────────────────────────────
    # Smaller log_mktcap = smaller company = higher score.
    # Negate so rank(pct=True) gives rank 1.0 to the smallest company.
    size_score = (-log_mktcap).rank(pct=True)
    size_score.name = "size_score"

    valid = size_score.notna().sum()
    logger.info(f"  Size scores computed for {valid} / {len(size_score)} tickers.")

    return size_score
