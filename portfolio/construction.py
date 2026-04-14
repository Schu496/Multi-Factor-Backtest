"""
portfolio/construction.py
--------------------------
This file answers one question: given our composite factor scores,
which stocks do we actually buy?

The answer is straightforward: we take the top N stocks by composite
score, where N is defined in configs/strategy.yaml (default: 50).

Think of it like a fantasy sports draft. Our factor model has ranked
every player (stock) in the league (S&P 500) from best to worst.
This file picks the top 50 players for our team.

We also handle some practical edge cases:
  - What if fewer than N stocks have valid scores? (Take what we have)
  - What if the same ticker appears twice? (Deduplicate)
  - What if the top score is suspiciously extreme? (Log a warning)

The output is a clean list of ticker symbols ready to be weighted
and handed to the backtest engine.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_portfolio(
    composite_scores: pd.Series,
    portfolio_size:   int = 50,
    min_score:        float = None,
) -> list[str]:
    """
    Selects the top N stocks by composite factor score.

    Parameters
    ----------
    composite_scores : pd.Series
        Output from composite.py. Index = tickers, values = scores.
        Should already be sorted highest to lowest.
    portfolio_size : int
        How many stocks to include. From strategy.yaml (default 50).
    min_score : float, optional
        If set, only include stocks with a score above this threshold.
        Useful for excluding stocks with very poor overall scores.
        Default is None (no minimum — just take the top N).

    Returns
    -------
    list[str]
        List of selected ticker symbols, ordered best score first.
    """

    logger.info(f"Selecting top {portfolio_size} stocks from {composite_scores.notna().sum()} ranked...")

    # ── Step 1: Remove stocks with no score ───────────────────────────────
    # NaN means we could not compute a reliable score for this stock.
    # We exclude them rather than risk selecting a stock by accident.
    valid_scores = composite_scores.dropna()

    if len(valid_scores) == 0:
        logger.error("  No valid scores found — cannot construct portfolio.")
        return []

    # ── Step 2: Sort highest to lowest ────────────────────────────────────
    # This should already be sorted from composite.py, but we sort again
    # as a safety net to guarantee correct ordering.
    valid_scores = valid_scores.sort_values(ascending=False)

    # ── Step 3: Apply minimum score filter (optional) ─────────────────────
    # If a minimum score threshold is set, exclude stocks below it.
    # This prevents us from buying a stock that scores well only because
    # everything else is terrible — a "least bad" problem.
    if min_score is not None:
        before = len(valid_scores)
        valid_scores = valid_scores[valid_scores >= min_score]
        after  = len(valid_scores)
        if before != after:
            logger.info(f"  Filtered {before - after} stocks below min score {min_score:.2f}")

    # ── Step 4: Handle case where fewer stocks are available than requested
    # This can happen early in the backtest when data is sparse, or if
    # many stocks failed the min_score filter.
    actual_size = min(portfolio_size, len(valid_scores))
    if actual_size < portfolio_size:
        logger.warning(
            f"  Only {actual_size} stocks available — "
            f"requested {portfolio_size}. Using {actual_size}."
        )

    if actual_size == 0:
        logger.error("  Portfolio is empty after filtering.")
        return []

    # ── Step 5: Select the top N ──────────────────────────────────────────
    selected = valid_scores.head(actual_size)

    # ── Step 6: Sanity checks ──────────────────────────────────────────────

    # Check for duplicate tickers (should never happen, but let's be safe)
    if selected.index.duplicated().any():
        logger.warning("  Duplicate tickers found — deduplicating.")
        selected = selected[~selected.index.duplicated(keep="first")]

    # Log the selected portfolio for transparency
    logger.info(f"\n  Portfolio selected: {len(selected)} stocks")
    logger.info(f"  Score range: {selected.iloc[0]:.3f} (best) to {selected.iloc[-1]:.3f} (worst in portfolio)")
    logger.info(f"\n  Selected stocks (top 10 shown):")
    for i, (ticker, score) in enumerate(selected.head(10).items()):
        logger.info(f"    {i+1:>2}. {ticker:<8} score: {score:+.3f}")
    if len(selected) > 10:
        logger.info(f"    ... and {len(selected) - 10} more")

    return selected.index.tolist()


def get_portfolio_turnover(
    previous_holdings: list[str],
    new_holdings:      list[str],
) -> dict:
    """
    Calculates how much the portfolio changed between two rebalance dates.

    "Turnover" is important because every trade costs money (bid-ask spread,
    commissions). High turnover eats into returns.

    Parameters
    ----------
    previous_holdings : list of tickers from last rebalance
    new_holdings      : list of tickers for this rebalance

    Returns
    -------
    dict with keys:
        added    — stocks newly entering the portfolio
        removed  — stocks leaving the portfolio
        retained — stocks staying in the portfolio
        turnover_pct — what fraction of the portfolio changed
    """

    prev_set = set(previous_holdings)
    new_set  = set(new_holdings)

    added    = sorted(new_set  - prev_set)   # in new but not old
    removed  = sorted(prev_set - new_set)    # in old but not new
    retained = sorted(prev_set & new_set)    # in both

    # Turnover = number of positions changed / total portfolio size
    # Each "add" and each "remove" counts as a change
    n_total  = max(len(prev_set), len(new_set), 1)
    turnover = len(added) / n_total  # adds = buys = half of two-sided turnover

    logger.info(f"  Portfolio turnover:")
    logger.info(f"    Retained : {len(retained)} stocks")
    logger.info(f"    Added    : {len(added)} stocks  → {added[:5]}{'...' if len(added)>5 else ''}")
    logger.info(f"    Removed  : {len(removed)} stocks → {removed[:5]}{'...' if len(removed)>5 else ''}")
    logger.info(f"    Turnover : {turnover:.1%} of portfolio")

    return {
        "added":        added,
        "removed":      removed,
        "retained":     retained,
        "turnover_pct": turnover,
    }
