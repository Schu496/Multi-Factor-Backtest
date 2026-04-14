"""
factors/value.py
----------------
The VALUE factor identifies stocks that are "cheap" relative to their
fundamental worth.

The core idea: if two companies have identical businesses but one trades
at half the price relative to its earnings or assets, the cheaper one
is more attractive. Over time, cheap stocks have historically outperformed
expensive ones — this is one of the most well-documented effects in finance.

We measure "cheapness" using three metrics:
  1. P/B ratio    — Price divided by Book Value (assets minus liabilities)
  2. P/E ratio    — Price divided by annual Earnings per share
  3. EV/EBITDA    — Enterprise Value divided by operating earnings

For all three: LOWER = CHEAPER = BETTER score.
We flip the sign so that a high score always means "good" for consistency.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_value_scores(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Computes a composite value score for each stock.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        The fundamentals DataFrame from data/ingest.py.
        Must contain columns: pb_ratio, pe_ratio, ev_ebitda.

    Returns
    -------
    pd.Series
        One score per ticker. Higher score = more attractive on value.
        Index is the ticker symbol.
    """

    logger.info("Computing value factor scores...")

    # ── Step 1: Extract the three value metrics ───────────────────────────
    # We make a copy so we don't accidentally modify the original data.
    df = fundamentals[["pb_ratio", "pe_ratio", "ev_ebitda"]].copy()

    # ── Step 2: Remove clearly bad data ───────────────────────────────────
    # Negative P/E ratios mean the company is losing money.
    # Negative P/B ratios mean the company has negative book value (rare but messy).
    # We set these to NaN (missing) so they don't pollute our rankings.
    df["pe_ratio"]  = df["pe_ratio"].where(df["pe_ratio"]  > 0)
    df["pb_ratio"]  = df["pb_ratio"].where(df["pb_ratio"]  > 0)
    df["ev_ebitda"] = df["ev_ebitda"].where(df["ev_ebitda"] > 0)

    # ── Step 3: Flip the sign ─────────────────────────────────────────────
    # Lower P/B, P/E, EV/EBITDA = cheaper = BETTER.
    # But our composite scorer expects: higher score = better.
    # Solution: multiply by -1. Now the cheapest stock gets the highest score.
    df["pb_score"]  = -df["pb_ratio"]
    df["pe_score"]  = -df["pe_ratio"]
    df["eveb_score"]= -df["ev_ebitda"]

    # ── Step 4: Rank each metric (0 to 1 scale) ───────────────────────────
    # We convert raw numbers to percentile ranks.
    # This makes the metrics comparable to each other before weighting.
    # pct=True means rank 1.0 = best, 0.0 = worst.
    scored = pd.DataFrame(index=df.index)
    scored["pb_rank"]   = df["pb_score"].rank(pct=True)
    scored["pe_rank"]   = df["pe_score"].rank(pct=True)
    scored["eveb_rank"] = df["eveb_score"].rank(pct=True)

    # ── Step 5: Weighted combination ──────────────────────────────────────
    # Weights from strategy.yaml: P/B=40%, EV/EBITDA=35%, P/E=25%
    # We use np.average which handles NaN gracefully via masking.
    def weighted_avg(row):
        vals    = [row["pb_rank"],   row["pe_rank"],   row["eveb_rank"]]
        weights = [0.40,             0.25,             0.35            ]
        # Only include metrics where we have data (not NaN)
        pairs = [(v, w) for v, w in zip(vals, weights) if pd.notna(v)]
        if not pairs:
            return np.nan
        v_arr, w_arr = zip(*pairs)
        return np.average(v_arr, weights=w_arr)

    composite = scored.apply(weighted_avg, axis=1)
    composite.name = "value_score"

    # ── Report ─────────────────────────────────────────────────────────────
    valid = composite.notna().sum()
    logger.info(f"  Value scores computed for {valid} / {len(composite)} tickers.")

    return composite
