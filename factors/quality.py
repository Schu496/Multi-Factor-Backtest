"""
factors/quality.py
------------------
The QUALITY factor identifies companies that are genuinely healthy and
profitable businesses — not just cheap or trending upward.

The intuition: a company that consistently earns high returns on its
capital, has wide profit margins, and doesn't rely heavily on debt is a
stronger business than one that looks similar on the surface but is
financially fragile. Quality companies tend to be more resilient during
market downturns and compound wealth more reliably over time.

We measure quality using three metrics:

  1. ROE (Return on Equity)
     How much profit does the company generate for every $1 of shareholder
     equity? A 20% ROE means $0.20 of profit per $1 invested. Higher = better.

  2. Gross Profit Margin
     What percentage of revenue remains after paying for the cost of goods?
     A 60% gross margin means $0.60 of every $1 in sales is kept.
     Higher margin = more pricing power = better business. Higher = better.

  3. Debt-to-Equity Ratio
     How much debt does the company carry relative to its equity?
     High debt means fragile finances and interest payments that eat profits.
     Lower = safer = better. (We flip the sign for scoring.)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_quality_scores(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Computes a composite quality score for each stock.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        The fundamentals DataFrame from data/ingest.py.
        Must contain columns: roe, gross_margin, debt_to_equity.

    Returns
    -------
    pd.Series
        One score per ticker. Higher score = higher quality company.
        Index is the ticker symbol.
    """

    logger.info("Computing quality factor scores...")

    df = fundamentals[["roe", "gross_margin", "debt_to_equity"]].copy()

    # ── Step 1: Clean bad data ────────────────────────────────────────────
    # ROE can be negative (company is losing money) — we keep negatives
    # because they are meaningful (very negative ROE = bad quality).
    # Gross margin should be between -1 and 1 (it's a percentage).
    # Debt-to-equity can be very large but not negative in a meaningful way.
    df["gross_margin"]   = df["gross_margin"].where(
        (df["gross_margin"] >= -1) & (df["gross_margin"] <= 1)
    )
    df["debt_to_equity"] = df["debt_to_equity"].where(df["debt_to_equity"] >= 0)

    # ── Step 2: Flip debt sign ────────────────────────────────────────────
    # Lower debt = better, but we need higher score = better.
    # So we negate it: a company with 0 debt gets the highest debt score.
    df["debt_score"] = -df["debt_to_equity"]

    # ── Step 3: Rank each metric ──────────────────────────────────────────
    scored = pd.DataFrame(index=df.index)
    scored["roe_rank"]    = df["roe"].rank(pct=True)           # higher ROE = better
    scored["margin_rank"] = df["gross_margin"].rank(pct=True)  # higher margin = better
    scored["debt_rank"]   = df["debt_score"].rank(pct=True)    # less debt = better

    # ── Step 4: Weighted combination ──────────────────────────────────────
    # Weights from strategy.yaml: ROE=40%, Gross Margin=35%, Debt/Equity=25%
    def weighted_avg(row):
        vals    = [row["roe_rank"], row["margin_rank"], row["debt_rank"]]
        weights = [0.40,            0.35,               0.25            ]
        pairs = [(v, w) for v, w in zip(vals, weights) if pd.notna(v)]
        if not pairs:
            return np.nan
        v_arr, w_arr = zip(*pairs)
        return np.average(v_arr, weights=w_arr)

    composite = scored.apply(weighted_avg, axis=1)
    composite.name = "quality_score"

    valid = composite.notna().sum()
    logger.info(f"  Quality scores computed for {valid} / {len(composite)} tickers.")

    return composite
