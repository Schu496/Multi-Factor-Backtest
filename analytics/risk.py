"""
analytics/risk.py
-----------------
This file goes deeper into risk analysis — beyond the single summary
numbers in performance.py, we look at how risk EVOLVED over time.

A strategy might have a great average Sharpe ratio, but if most of
that came from 2010-2015 and the strategy fell apart after that,
you would want to know. Rolling metrics reveal this.

We also calculate tail risk metrics:

  Value at Risk (VaR)
    "On a typical bad day, how much could we lose?"
    95% VaR of -1.5% means: on 95% of days the loss is less than 1.5%.
    Put another way: there is a 5% chance of losing more than 1.5% in a day.

  Conditional VaR (CVaR) / Expected Shortfall
    "On the really bad days — the worst 5% — how much do we lose on average?"
    Always worse than VaR by definition. Gives a better picture of
    catastrophic risk because it looks at the average of all the tail losses,
    not just the threshold.

Rolling metrics use a sliding window (e.g. 252 trading days = 1 year)
and compute the metric for each window. The result is a time series
showing how the metric changed throughout the backtest.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

RISK_FREE_DAILY = 0.03 / 252


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk at the given confidence level.

    Parameters
    ----------
    returns    : daily return series
    confidence : e.g. 0.95 means "95% of days are better than this"

    Returns
    -------
    float : negative number, e.g. -0.015 means -1.5% daily VaR

    Method: historical simulation — we just look at the actual
    distribution of past returns and find the percentile.
    No assumptions about normal distributions needed.
    """
    if len(returns) < 20:
        return np.nan

    # The (1-confidence) percentile = the loss threshold
    # e.g. 5th percentile for 95% confidence
    var = returns.quantile(1 - confidence)
    return var


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).

    The average return on the worst (1-confidence)% of days.
    Always more negative than VaR — it is the average of the tail,
    not just the edge of it.

    Example: if VaR is -1.5%, CVaR might be -2.3%, meaning the
    average loss on a "tail risk day" is 2.3%.
    """
    if len(returns) < 20:
        return np.nan

    var         = compute_var(returns, confidence)
    tail_losses = returns[returns <= var]

    if len(tail_losses) == 0:
        return var

    cvar = tail_losses.mean()
    return cvar


def compute_rolling_volatility(
    returns: pd.Series,
    window:  int = 63,    # ~3 months of trading days
) -> pd.Series:
    """
    Annualized volatility computed over a rolling window.

    Shows how the strategy's risk level changed over time.
    Spikes in rolling vol correspond to turbulent market periods
    (e.g. 2020 COVID crash, 2022 rate hike regime).
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_vol.name = "rolling_volatility"
    return rolling_vol


def compute_rolling_sharpe(
    returns: pd.Series,
    window:  int = 252,   # 1 year of trading days
) -> pd.Series:
    """
    Sharpe ratio computed over a rolling 1-year window.

    Shows whether the strategy's risk-adjusted performance was
    consistent or concentrated in specific periods.

    A strategy with rolling Sharpe that is always positive is
    much more reliable than one that is sometimes 3.0 and sometimes -1.0.
    """
    excess = returns - RISK_FREE_DAILY

    rolling_mean = excess.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    rolling_sharpe.name = "rolling_sharpe"
    return rolling_sharpe


def compute_rolling_drawdown(returns: pd.Series) -> pd.Series:
    """
    Drawdown at every point in time — not just the maximum.

    Shows the full history of how far below the high-water mark
    the portfolio was at each date. Useful for visualizing
    recovery periods after losses.

    Returns a series of values between 0 and -1.
    0 = at all-time high
    -0.20 = 20% below the all-time high
    """
    cumulative  = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown    = (cumulative - running_max) / running_max
    drawdown.name = "drawdown"
    return drawdown


def compute_drawdown_periods(returns: pd.Series) -> pd.DataFrame:
    """
    Identifies and ranks all significant drawdown periods.

    For each drawdown event, records:
      - Start date (when we fell from the peak)
      - Trough date (when the loss was worst)
      - End date (when we recovered to the previous peak)
      - Max loss during the period
      - Duration in trading days

    Returns the top 10 worst drawdowns sorted by severity.
    """
    cumulative    = (1 + returns).cumprod()
    running_max   = cumulative.cummax()
    drawdown      = (cumulative - running_max) / running_max

    # Find where drawdowns start and end
    in_drawdown = drawdown < 0
    periods     = []
    start       = None

    for date, val in in_drawdown.items():
        if val and start is None:
            start = date                          # drawdown begins
        elif not val and start is not None:
            # Drawdown ended — record this period
            period_dd = drawdown.loc[start:date]
            trough    = period_dd.idxmin()
            max_loss  = period_dd.min()
            duration  = (date - start).days

            periods.append({
                "start":     start,
                "trough":    trough,
                "end":       date,
                "max_loss":  max_loss,
                "duration_days": duration,
            })
            start = None

    # Handle case where we end the backtest still in a drawdown
    if start is not None:
        period_dd = drawdown.loc[start:]
        trough    = period_dd.idxmin()
        max_loss  = period_dd.min()
        periods.append({
            "start":         start,
            "trough":        trough,
            "end":           None,       # never recovered
            "max_loss":      max_loss,
            "duration_days": None,
        })

    if not periods:
        return pd.DataFrame()

    df = pd.DataFrame(periods)
    df = df.sort_values("max_loss").head(10)   # top 10 worst
    df = df.reset_index(drop=True)
    df.index += 1   # rank starts at 1

    return df


def compute_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Builds a Year × Month table of monthly returns.

    This is the classic "calendar returns" table you see in hedge fund
    tear sheets. Each cell shows the return for that specific month.

    Example (partial):
             Jan    Feb    Mar  ...  Dec   Full Year
    2010   +2.1%  -0.8%  +3.2%     +1.1%   +15.3%
    2011   -1.2%  +0.5%  -2.1%     -0.3%    -5.8%
    ...
    """
    # Resample to monthly returns
    monthly = (1 + returns).resample("ME").prod() - 1

    # Build a pivot table: rows=years, columns=months
    df = pd.DataFrame({
        "year":   monthly.index.year,
        "month":  monthly.index.month,
        "return": monthly.values,
    })

    table = df.pivot(index="year", columns="month", values="return")

    # Rename columns from month numbers to abbreviations
    month_names = {
        1: "Jan", 2: "Feb",  3: "Mar",  4: "Apr",
        5: "May", 6: "Jun",  7: "Jul",  8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    table = table.rename(columns=month_names)

    # Add a full-year column
    annual = (1 + returns).resample("YE").prod() - 1
    table["Full Year"] = annual.values[:len(table)]

    return table


def compute_all_risk_metrics(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Master function — computes all risk metrics at once.

    Returns a dict containing both scalar metrics and time series.
    """

    logger.info("Computing risk metrics...")

    # Align series
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat   = aligned.iloc[:, 0]
    bench   = aligned.iloc[:, 1]

    results = {
        # ── Tail risk (scalar) ──────────────────────────────────────
        "var_95":               compute_var(strat, 0.95),
        "var_99":               compute_var(strat, 0.99),
        "cvar_95":              compute_cvar(strat, 0.95),
        "cvar_99":              compute_cvar(strat, 0.99),
        "benchmark_var_95":     compute_var(bench, 0.95),
        "benchmark_cvar_95":    compute_cvar(bench, 0.95),

        # ── Rolling time series ─────────────────────────────────────
        "rolling_vol_63":       compute_rolling_volatility(strat, 63),
        "rolling_vol_252":      compute_rolling_volatility(strat, 252),
        "rolling_sharpe_252":   compute_rolling_sharpe(strat, 252),
        "drawdown_series":      compute_rolling_drawdown(strat),
        "benchmark_drawdown":   compute_rolling_drawdown(bench),

        # ── Drawdown analysis ───────────────────────────────────────
        "top_drawdowns":        compute_drawdown_periods(strat),

        # ── Monthly returns calendar ────────────────────────────────
        "monthly_returns_table": compute_monthly_returns_table(strat),
    }

    # Log tail risk summary
    logger.info(f"  VaR  (95%): {results['var_95']:.2%} daily")
    logger.info(f"  CVaR (95%): {results['cvar_95']:.2%} daily")
    logger.info(f"  VaR  (99%): {results['var_99']:.2%} daily")

    if not results["top_drawdowns"].empty:
        worst = results["top_drawdowns"].iloc[0]
        logger.info(f"  Worst drawdown: {worst['max_loss']:.2%} "
                    f"(started {worst['start'].date()})")

    return results
