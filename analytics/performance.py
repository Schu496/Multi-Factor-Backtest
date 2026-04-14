"""
analytics/performance.py
-------------------------
This file calculates the key performance metrics that tell us whether
our strategy is actually good.

Anyone can make money in a bull market by just buying everything.
The question is: did our strategy make MORE money than simply buying
the S&P 500 index (SPY)? And did it do so without taking on extra risk?

These metrics answer exactly that. Here is a plain-English guide to
each one:

  CAGR (Compound Annual Growth Rate)
    The single annualized return number. "This strategy returned 12%
    per year on average." Accounts for compounding.

  Sharpe Ratio
    Return per unit of risk. A Sharpe of 1.0 means you earned 1% of
    excess return for every 1% of volatility. Higher is better.
    Above 1.0 is good. Above 2.0 is excellent. Below 0.5 is poor.

  Sortino Ratio
    Like Sharpe, but only penalizes DOWNSIDE volatility. Going up
    fast is not "risk" — only going down is. More nuanced than Sharpe.

  Maximum Drawdown
    The worst peak-to-trough loss experienced. If the portfolio hit
    $1.5M then fell to $1.2M before recovering, that is a -20% drawdown.
    Tells you the worst pain an investor would have felt.

  Calmar Ratio
    CAGR divided by Maximum Drawdown. Measures return per unit of
    drawdown risk. A Calmar of 0.5 means you earned 0.5% of annual
    return for every 1% of max drawdown.

  Information Ratio
    How consistently did we beat the benchmark? IR = (strategy return
    minus benchmark return) / tracking error. Higher = more consistent
    outperformance.

  Win Rate
    What percentage of months did the strategy have positive returns?
    A strategy can have a low win rate but still be profitable if the
    wins are much bigger than the losses.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Risk-free rate assumption — approximate average US 3-month T-bill rate
# Used in Sharpe and Sortino calculations
RISK_FREE_RATE_ANNUAL = 0.03   # 3% per year
RISK_FREE_RATE_DAILY  = RISK_FREE_RATE_ANNUAL / 252


def compute_cagr(returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    Converts total return into an annualized percentage.

    Formula: (1 + total_return) ^ (1 / years) - 1

    Example: portfolio doubled in 6 years → CAGR = 2^(1/6) - 1 = 12.2%
    """
    if len(returns) < 2:
        return np.nan

    total_return = (1 + returns).prod() - 1
    years        = len(returns) / 252    # 252 trading days per year
    cagr         = (1 + total_return) ** (1 / years) - 1
    return cagr


def compute_sharpe(returns: pd.Series) -> float:
    """
    Sharpe Ratio — return per unit of total volatility.

    Formula: (mean_daily_return - risk_free_daily) / std_daily × √252

    The √252 annualizes the ratio (converts from daily to annual scale).
    """
    if len(returns) < 20:
        return np.nan

    excess_returns = returns - RISK_FREE_RATE_DAILY
    mean_excess    = excess_returns.mean()
    std_returns    = returns.std()

    if std_returns == 0:
        return np.nan

    sharpe = (mean_excess / std_returns) * np.sqrt(252)
    return sharpe


def compute_sortino(returns: pd.Series) -> float:
    """
    Sortino Ratio — return per unit of DOWNSIDE volatility only.

    Only days with negative returns count as "risk."
    Upward volatility (good days) is not penalized.

    Formula: (mean_return - risk_free) / downside_std × √252
    """
    if len(returns) < 20:
        return np.nan

    excess_returns  = returns - RISK_FREE_RATE_DAILY
    mean_excess     = excess_returns.mean()

    # Only keep negative returns for downside deviation
    downside        = returns[returns < 0]
    if len(downside) == 0:
        return np.inf   # no losing days — technically infinite Sortino

    downside_std    = downside.std()
    if downside_std == 0:
        return np.nan

    sortino = (mean_excess / downside_std) * np.sqrt(252)
    return sortino


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown — the worst peak-to-trough decline.

    We track the portfolio's "high water mark" (the highest value ever
    reached) and measure how far it fell from that peak at any point.

    Returns a negative number (e.g. -0.35 means a 35% drawdown).
    """
    if len(returns) < 2:
        return np.nan

    # Build the cumulative wealth index (starts at 1.0)
    cumulative = (1 + returns).cumprod()

    # Running maximum — the highest level reached up to each date
    running_max = cumulative.cummax()

    # Drawdown at each point = how far below the peak are we right now
    drawdown    = (cumulative - running_max) / running_max

    max_dd = drawdown.min()   # most negative value = worst drawdown
    return max_dd


def compute_calmar(returns: pd.Series) -> float:
    """
    Calmar Ratio — CAGR divided by absolute maximum drawdown.

    A ratio of 1.0 means the strategy earns 1% of annual return
    for every 1% of maximum drawdown experienced.
    Higher is better.
    """
    cagr   = compute_cagr(returns)
    max_dd = compute_max_drawdown(returns)

    if max_dd == 0 or pd.isna(max_dd):
        return np.nan

    return cagr / abs(max_dd)


def compute_information_ratio(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information Ratio — consistency of outperformance vs benchmark.

    Active return  = strategy return - benchmark return (each day)
    Tracking error = standard deviation of active returns
    IR             = mean(active return) / tracking error × √252

    IR > 0.5  is good
    IR > 1.0  is excellent
    IR < 0    means we underperformed the benchmark consistently
    """
    # Align both series to the same dates
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    active_returns = strat - bench
    mean_active    = active_returns.mean()
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return np.nan

    ir = (mean_active / tracking_error) * np.sqrt(252)
    return ir


def compute_win_rate(returns: pd.Series, freq: str = "M") -> float:
    """
    Win Rate — percentage of periods with positive returns.

    Parameters
    ----------
    returns : daily return series
    freq    : 'D' for daily, 'M' for monthly (default), 'Y' for annual

    Returns the fraction of periods that were profitable (0.0 to 1.0).
    A win rate of 0.60 means 60% of months were positive.
    """
    if freq != "D":
        # Resample to the requested frequency
        period_returns = (1 + returns).resample(freq).prod() - 1
    else:
        period_returns = returns

    wins     = (period_returns > 0).sum()
    total    = period_returns.notna().sum()

    if total == 0:
        return np.nan

    return wins / total


def compute_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility — standard deviation of daily returns × √252.

    A volatility of 15% means the portfolio moves roughly ±15% per year
    in a "normal" environment (one standard deviation).
    """
    return returns.std() * np.sqrt(252)


def compute_beta(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta — how much the strategy moves relative to the market.

    Beta of 1.0 = moves exactly with the market
    Beta of 0.8 = moves 80% as much as the market (defensive)
    Beta of 1.2 = moves 120% as much as the market (aggressive)
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    cov = strat.cov(bench)
    var = bench.var()

    if var == 0:
        return np.nan

    return cov / var


def compute_alpha(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Alpha — the return the strategy generated BEYOND what the market
    exposure alone would explain.

    Alpha = strategy_return - (risk_free + beta × (market - risk_free))

    Positive alpha means the factor selection genuinely added value.
    """
    beta  = compute_beta(strategy_returns, benchmark_returns)
    if pd.isna(beta):
        return np.nan

    strat_cagr = compute_cagr(strategy_returns)
    bench_cagr = compute_cagr(benchmark_returns)

    alpha = strat_cagr - (RISK_FREE_RATE_ANNUAL + beta * (bench_cagr - RISK_FREE_RATE_ANNUAL))
    return alpha


def compute_all_metrics(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Master function — computes every performance metric at once.

    Parameters
    ----------
    strategy_returns  : daily returns of our factor portfolio
    benchmark_returns : daily returns of SPY benchmark

    Returns
    -------
    dict with all metrics, ready for the reporting layer to display.
    """

    logger.info("Computing performance metrics...")

    # Align to matching dates
    aligned   = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat     = aligned.iloc[:, 0]
    bench     = aligned.iloc[:, 1]

    metrics = {
        # ── Return metrics ─────────────────────────────────────────
        "cagr":               compute_cagr(strat),
        "benchmark_cagr":     compute_cagr(bench),
        "total_return":       (1 + strat).prod() - 1,
        "benchmark_total":    (1 + bench).prod() - 1,

        # ── Risk metrics ───────────────────────────────────────────
        "volatility":         compute_volatility(strat),
        "benchmark_vol":      compute_volatility(bench),
        "max_drawdown":       compute_max_drawdown(strat),
        "benchmark_max_dd":   compute_max_drawdown(bench),

        # ── Risk-adjusted return metrics ───────────────────────────
        "sharpe":             compute_sharpe(strat),
        "benchmark_sharpe":   compute_sharpe(bench),
        "sortino":            compute_sortino(strat),
        "calmar":             compute_calmar(strat),

        # ── Relative metrics ───────────────────────────────────────
        "information_ratio":  compute_information_ratio(strat, bench),
        "beta":               compute_beta(strat, bench),
        "alpha":              compute_alpha(strat, bench),

        # ── Win/loss metrics ───────────────────────────────────────
        "monthly_win_rate":   compute_win_rate(strat, "ME"),
        "daily_win_rate":     compute_win_rate(strat, "D"),
    }

    # ── Log a clean summary ────────────────────────────────────────────
    logger.info("\n" + "═" * 50)
    logger.info("  PERFORMANCE SUMMARY")
    logger.info("═" * 50)
    logger.info(f"  {'Metric':<25} {'Strategy':>10}  {'Benchmark':>10}")
    logger.info(f"  {'-'*25} {'-'*10}  {'-'*10}")
    logger.info(f"  {'CAGR':<25} {metrics['cagr']:>10.2%}  {metrics['benchmark_cagr']:>10.2%}")
    logger.info(f"  {'Total Return':<25} {metrics['total_return']:>10.2%}  {metrics['benchmark_total']:>10.2%}")
    logger.info(f"  {'Volatility':<25} {metrics['volatility']:>10.2%}  {metrics['benchmark_vol']:>10.2%}")
    logger.info(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>10.2%}  {metrics['benchmark_max_dd']:>10.2%}")
    logger.info(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>10.2f}  {metrics['benchmark_sharpe']:>10.2f}")
    logger.info(f"  {'Sortino Ratio':<25} {metrics['sortino']:>10.2f}  {'—':>10}")
    logger.info(f"  {'Calmar Ratio':<25} {metrics['calmar']:>10.2f}  {'—':>10}")
    logger.info(f"  {'Information Ratio':<25} {metrics['information_ratio']:>10.2f}  {'—':>10}")
    logger.info(f"  {'Beta':<25} {metrics['beta']:>10.2f}  {'1.00':>10}")
    logger.info(f"  {'Alpha (annual)':<25} {metrics['alpha']:>10.2%}  {'—':>10}")
    logger.info(f"  {'Monthly Win Rate':<25} {metrics['monthly_win_rate']:>10.2%}  {'—':>10}")
    logger.info("═" * 50)

    return metrics
