"""
reporting/charts.py
--------------------
This file generates all the individual charts that go into the tearsheet.

Each function creates one chart and saves it as a PNG file to
reporting/output/. The tearsheet.py file then assembles them all
into a single PDF.

We use matplotlib for all charts — it is the most reliable library
for generating publication-quality static charts in Python.

Chart inventory:
  1. Cumulative returns     — strategy vs SPY growth of $1
  2. Drawdown chart         — how far below the peak at each date
  3. Rolling Sharpe         — 12-month rolling risk-adjusted return
  4. Monthly returns heatmap— calendar grid of monthly P&L
  5. Factor attribution     — bar chart of each factor's contribution
  6. Annual returns         — side-by-side bars strategy vs benchmark
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("reporting/output")

# ── Consistent color palette ──────────────────────────────────────────────────
STRATEGY_COLOR  = "#2563EB"   # blue  — our portfolio
BENCHMARK_COLOR = "#6B7280"   # gray  — SPY benchmark
LOSS_COLOR      = "#DC2626"   # red   — drawdowns and losses
PROFIT_COLOR    = "#16A34A"   # green — gains

# ── Chart style defaults ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.color":        "#D1D5DB",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})


def _save(fig: plt.Figure, filename: str) -> Path:
    """Saves a figure to the output directory and closes it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Cumulative Returns
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
    initial_capital:   float = 1_000_000,
) -> Path:
    """
    Shows the growth of $1,000,000 over the backtest period for both
    the strategy and the SPY benchmark.

    This is the most important chart — it shows at a glance whether
    the strategy outperformed and by how much.
    """

    logger.info("  Generating cumulative returns chart...")

    # Align both series to the same dates
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat   = aligned.iloc[:, 0]
    bench   = aligned.iloc[:, 1]

    # Build the wealth index (cumulative product of (1 + daily_return))
    strat_wealth = (1 + strat).cumprod()  * initial_capital
    bench_wealth = (1 + bench).cumprod()  * initial_capital

    # Compute final values for the legend
    strat_final  = strat_wealth.iloc[-1]
    bench_final  = bench_wealth.iloc[-1]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        strat_wealth.index, strat_wealth / 1e6,
        color=STRATEGY_COLOR, linewidth=2,
        label=f"Strategy  ${strat_final/1e6:.2f}M",
    )
    ax.plot(
        bench_wealth.index, bench_wealth / 1e6,
        color=BENCHMARK_COLOR, linewidth=1.5, linestyle="--",
        label=f"SPY       ${bench_final/1e6:.2f}M",
    )

    # Shade the area between the two lines
    ax.fill_between(
        strat_wealth.index,
        strat_wealth / 1e6,
        bench_wealth / 1e6,
        where  = strat_wealth >= bench_wealth,
        alpha  = 0.08,
        color  = STRATEGY_COLOR,
        label  = "Outperformance",
    )
    ax.fill_between(
        strat_wealth.index,
        strat_wealth / 1e6,
        bench_wealth / 1e6,
        where  = strat_wealth < bench_wealth,
        alpha  = 0.08,
        color  = LOSS_COLOR,
        label  = "Underperformance",
    )

    ax.set_title("Cumulative Returns — Strategy vs S&P 500 (SPY)")
    ax.set_ylabel("Portfolio Value ($M)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(strat_wealth.index[0], strat_wealth.index[-1])

    fig.tight_layout()
    return _save(fig, "01_cumulative_returns.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — Drawdown Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_drawdown(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> Path:
    """
    Shows how far the portfolio was below its all-time high at each date.

    The deeper and longer the red shading, the more painful the period
    was for an investor. A good strategy has shallow, short drawdowns.
    """

    logger.info("  Generating drawdown chart...")

    from analytics.risk import compute_rolling_drawdown

    strat_dd = compute_rolling_drawdown(strategy_returns)
    bench_dd = compute_rolling_drawdown(benchmark_returns)

    # Align
    aligned  = pd.concat([strat_dd, bench_dd], axis=1).dropna()
    strat_dd = aligned.iloc[:, 0]
    bench_dd = aligned.iloc[:, 1]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(
        strat_dd.index, strat_dd * 100, 0,
        color=STRATEGY_COLOR, alpha=0.4, label="Strategy drawdown"
    )
    ax.fill_between(
        bench_dd.index, bench_dd * 100, 0,
        color=BENCHMARK_COLOR, alpha=0.2, label="SPY drawdown"
    )
    ax.plot(strat_dd.index, strat_dd * 100, color=STRATEGY_COLOR, linewidth=0.8)
    ax.plot(bench_dd.index, bench_dd * 100, color=BENCHMARK_COLOR, linewidth=0.8, linestyle="--")

    ax.set_title("Drawdown from Peak")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(strat_dd.index[0], strat_dd.index[-1])

    # Annotate the maximum drawdown point
    min_idx = strat_dd.idxmin()
    min_val = strat_dd.min()
    ax.annotate(
        f"Max DD\n{min_val:.1%}",
        xy=(min_idx, min_val * 100),
        xytext=(30, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=LOSS_COLOR),
        color=LOSS_COLOR, fontsize=9,
    )

    fig.tight_layout()
    return _save(fig, "02_drawdown.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Rolling Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_sharpe(strategy_returns: pd.Series) -> Path:
    """
    Shows the 12-month rolling Sharpe ratio over time.

    A consistently positive rolling Sharpe means the strategy was
    reliably generating risk-adjusted returns throughout the period —
    not just during one lucky streak.
    """

    logger.info("  Generating rolling Sharpe chart...")

    from analytics.risk import compute_rolling_sharpe
    rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=252)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Color the line green above 0, red below 0
    ax.axhline(0,   color="#9CA3AF", linewidth=1.0, linestyle="--")
    ax.axhline(1,   color=PROFIT_COLOR, linewidth=0.8, linestyle=":", alpha=0.6, label="Sharpe = 1.0 (good)")
    ax.axhline(-0.5,color=LOSS_COLOR,   linewidth=0.8, linestyle=":", alpha=0.6, label="Sharpe = -0.5 (poor)")

    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe, 0,
        where  = rolling_sharpe >= 0,
        alpha  = 0.25, color=PROFIT_COLOR,
    )
    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe, 0,
        where  = rolling_sharpe < 0,
        alpha  = 0.25, color=LOSS_COLOR,
    )
    ax.plot(rolling_sharpe.index, rolling_sharpe, color=STRATEGY_COLOR, linewidth=1.5)

    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_xlim(rolling_sharpe.dropna().index[0], rolling_sharpe.index[-1])

    fig.tight_layout()
    return _save(fig, "03_rolling_sharpe.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Monthly Returns Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_monthly_returns_heatmap(strategy_returns: pd.Series) -> Path:
    """
    A calendar grid showing each month's return color-coded from red to green.

    This is the classic hedge fund tearsheet table. At a glance you can
    see which years and months were strong or weak.
    Green = positive month, Red = negative month.
    Darker color = more extreme return.
    """

    logger.info("  Generating monthly returns heatmap...")

    from analytics.risk import compute_monthly_returns_table
    table = compute_monthly_returns_table(strategy_returns)

    # Drop the "Full Year" column for the heatmap (we show it as text)
    annual_col = table["Full Year"] if "Full Year" in table.columns else None
    month_cols = [c for c in table.columns if c != "Full Year"]
    heat_data  = table[month_cols]

    n_years  = len(heat_data)
    n_months = len(month_cols)

    fig, ax = plt.subplots(figsize=(14, max(4, n_years * 0.55)))

    # Custom red-white-green colormap
    cmap = LinearSegmentedColormap.from_list(
        "rwg", [LOSS_COLOR, "white", PROFIT_COLOR], N=256
    )

    # Draw the heatmap manually using imshow
    data_array = heat_data.values * 100   # convert to percentage
    vmax = min(abs(np.nanpercentile(data_array, 95)), 10)   # cap color scale at ±10%
    im   = ax.imshow(data_array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # Add text labels inside each cell
    for i in range(n_years):
        for j in range(n_months):
            val = data_array[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold",
                )

    # Add annual return text on the right
    if annual_col is not None:
        for i, (year, ann_ret) in enumerate(annual_col.items()):
            color = PROFIT_COLOR if ann_ret >= 0 else LOSS_COLOR
            ax.text(
                n_months + 0.2, i, f"{ann_ret*100:+.1f}%",
                ha="left", va="center",
                fontsize=9, color=color, fontweight="bold",
            )

    # Axis labels
    ax.set_xticks(range(n_months))
    ax.set_xticklabels(month_cols, fontsize=9)
    ax.set_yticks(range(n_years))
    ax.set_yticklabels(heat_data.index.astype(str), fontsize=9)
    ax.set_title("Monthly Returns Heatmap (%)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    cbar.set_label("Return (%)", fontsize=9)

    fig.tight_layout()
    return _save(fig, "04_monthly_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Factor Attribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_attribution(attr_df: pd.DataFrame) -> Path:
    """
    Bar chart showing each factor's annualized return contribution.

    Tells you at a glance which factors added value (green bars)
    and which were a drag (red bars) over the backtest period.
    """

    logger.info("  Generating factor attribution chart...")

    if attr_df is None or attr_df.empty:
        logger.warning("  No attribution data — skipping chart.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Attribution data not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return _save(fig, "05_factor_attribution.png")

    # Annualize the mean monthly returns per factor
    ann_returns = {}
    for col in attr_df.columns:
        monthly_mean  = attr_df[col].dropna().mean()
        ann_returns[col] = (1 + monthly_mean) ** 12 - 1

    factors = list(ann_returns.keys())
    values  = [ann_returns[f] * 100 for f in factors]
    colors  = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in values]

    # Clean up factor names for display
    labels = [f.replace("_", " ").title() for f in factors]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:+.1f}%",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10, fontweight="bold",
            color=PROFIT_COLOR if val >= 0 else LOSS_COLOR,
        )

    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_title("Factor Attribution — Annualized Return by Factor")
    ax.set_ylabel("Annualized Return (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.tight_layout()
    return _save(fig, "05_factor_attribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 6 — Annual Returns Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_annual_returns(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> Path:
    """
    Side-by-side bar chart of annual returns for strategy vs SPY.

    Makes it easy to see which years the strategy beat the benchmark
    and which years it lagged.
    """

    logger.info("  Generating annual returns chart...")

    # Resample to annual returns
    strat_annual = (1 + strategy_returns).resample("YE").prod() - 1
    bench_annual = (1 + benchmark_returns).resample("YE").prod() - 1

    aligned = pd.concat([strat_annual, bench_annual], axis=1).dropna()
    years   = aligned.index.year.astype(str)
    strat_v = aligned.iloc[:, 0].values * 100
    bench_v = aligned.iloc[:, 1].values * 100

    x      = np.arange(len(years))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))

    # Strategy bars — green if positive, red if negative
    strat_colors = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in strat_v]
    bench_colors = [f"{BENCHMARK_COLOR}CC" for _ in bench_v]   # semi-transparent

    ax.bar(x - width/2, strat_v, width, color=strat_colors, alpha=0.85,
           label="Strategy",   edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, bench_v, width, color=bench_colors, alpha=0.60,
           label="SPY",        edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_title("Annual Returns — Strategy vs SPY")
    ax.set_ylabel("Annual Return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "06_annual_returns.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_charts(
    results:      dict,
    attr_df,
) -> dict:
    """
    Generates all charts and returns a dict of {chart_name: file_path}.

    Parameters
    ----------
    results  : output dict from BacktestEngine.run()
    attr_df  : output DataFrame from compute_factor_attribution()

    Returns
    -------
    dict mapping chart names to their saved file paths.
    """

    logger.info("Generating all charts...")

    strat = results["portfolio_returns"]
    bench = results["benchmark_returns"]
    cap   = results.get("initial_capital", 1_000_000)

    chart_paths = {}

    try:
        chart_paths["cumulative"]   = plot_cumulative_returns(strat, bench, cap)
    except Exception as e:
        logger.warning(f"  Cumulative returns chart failed: {e}")

    try:
        chart_paths["drawdown"]     = plot_drawdown(strat, bench)
    except Exception as e:
        logger.warning(f"  Drawdown chart failed: {e}")

    try:
        chart_paths["rolling_sharpe"] = plot_rolling_sharpe(strat)
    except Exception as e:
        logger.warning(f"  Rolling Sharpe chart failed: {e}")

    try:
        chart_paths["monthly_heatmap"] = plot_monthly_returns_heatmap(strat)
    except Exception as e:
        logger.warning(f"  Monthly heatmap failed: {e}")

    try:
        chart_paths["attribution"]  = plot_factor_attribution(attr_df)
    except Exception as e:
        logger.warning(f"  Attribution chart failed: {e}")

    try:
        chart_paths["annual"]       = plot_annual_returns(strat, bench)
    except Exception as e:
        logger.warning(f"  Annual returns chart failed: {e}")

    logger.info(f"  {len(chart_paths)}/6 charts generated successfully.")
    return chart_paths
