"""
reporting/tearsheet.py
-----------------------
This file assembles everything into a single professional PDF report —
the "tearsheet."

A tearsheet is the standard output format in the hedge fund and asset
management world. When a portfolio manager presents a strategy to
investors or to an investment committee, they hand out a tearsheet.
It fits the entire story of a strategy onto a small number of pages:

  Page 1 — Summary statistics table
            The key numbers at a glance: CAGR, Sharpe, drawdown, etc.
            Strategy vs benchmark side by side.

  Page 2 — Cumulative returns + drawdown
            The two most important visual charts together.

  Page 3 — Rolling metrics + annual returns
            Shows consistency over time and year-by-year breakdown.

  Page 4 — Monthly returns heatmap + factor attribution
            Granular monthly detail and which factors drove performance.

We use matplotlib's PdfPages to stitch everything into one file.
The final PDF is saved to reporting/output/tearsheet.pdf.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reporting/output")

# ── Color palette (matches charts.py) ─────────────────────────────────────────
STRATEGY_COLOR  = "#2563EB"
BENCHMARK_COLOR = "#6B7280"
LOSS_COLOR      = "#DC2626"
PROFIT_COLOR    = "#16A34A"
DARK             = "#111827"
LIGHT_GRAY       = "#F9FAFB"
MID_GRAY         = "#E5E7EB"


class TearsheetGenerator:
    """
    Generates a professional multi-page PDF tearsheet.

    Usage in main.py:
        generator = TearsheetGenerator(
            config, perf_metrics, risk_metrics, attr_df, results
        )
        generator.generate()
    """

    def __init__(
        self,
        config:       dict,
        perf_metrics: dict,
        risk_metrics: dict,
        attr_df,
        results:      dict,
    ):
        self.config       = config
        self.perf         = perf_metrics
        self.risk         = risk_metrics
        self.attr_df      = attr_df
        self.results      = results
        self.strat        = results["portfolio_returns"]
        self.bench        = results["benchmark_returns"]
        self.cap          = results.get("initial_capital", 1_000_000)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    def generate(self) -> Path:
        """
        Generates the full tearsheet PDF.
        Returns the path to the saved file.
        """

        output_path = OUTPUT_DIR / "tearsheet.pdf"

        logger.info("=" * 60)
        logger.info("  GENERATING TEARSHEET")
        logger.info("=" * 60)

        with PdfPages(output_path) as pdf:

            logger.info("  Page 1: Summary statistics...")
            fig = self._make_summary_page()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            logger.info("  Page 2: Returns and drawdown...")
            fig = self._make_returns_page()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            logger.info("  Page 3: Rolling metrics and annual returns...")
            fig = self._make_rolling_page()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            logger.info("  Page 4: Monthly heatmap and factor attribution...")
            fig = self._make_detail_page()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # PDF metadata
            d = pdf.infodict()
            d["Title"]   = "Multi-Factor Equity Strategy Tearsheet"
            d["Author"]  = "Multi-Factor Backtest"
            d["Subject"] = f"S&P 500 Factor Strategy {self.config['start_date']} to {self.config['end_date']}"

        logger.info(f"  Tearsheet saved to: {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────
    def _header(self, fig, title: str):
        """Adds a consistent header bar to each page."""
        fig.text(
            0.04, 0.97, title,
            fontsize=16, fontweight="bold", color=DARK,
            va="top",
        )
        fig.text(
            0.96, 0.97,
            f"Generated {datetime.today().strftime('%B %d, %Y')}",
            fontsize=9, color=BENCHMARK_COLOR,
            va="top", ha="right",
        )
        # Thin horizontal rule under the header
        line = plt.Line2D(
            [0.04, 0.96], [0.955, 0.955],
            transform=fig.transFigure,
            color=MID_GRAY, linewidth=1,
        )
        fig.add_artist(line)

    # ──────────────────────────────────────────────────────────────────────
    def _make_summary_page(self) -> plt.Figure:
        """
        Page 1 — the executive summary.

        Two side-by-side metric panels (strategy vs benchmark),
        plus a brief strategy description at the top.
        """

        fig = plt.figure(figsize=(11, 8.5))
        self._header(fig, "Multi-Factor Equity Strategy — Performance Summary")

        # ── Strategy description block ─────────────────────────────────
        desc = (
            f"Universe: S&P 500   |   Period: {self.config['start_date']} – {self.config['end_date']}   |   "
            f"Factors: Value · Momentum · Quality · Low Vol · Size   |   "
            f"Rebalance: {self.config['rebalance_freq'].title()}   |   "
            f"Portfolio: Top {self.config['portfolio_size']} stocks   |   "
            f"Weighting: {self.config['weighting'].replace('_', ' ').title()}   |   "
            f"Transaction cost: {self.config['transaction_cost_bps']} bps"
        )
        fig.text(0.04, 0.92, desc, fontsize=8.5, color=BENCHMARK_COLOR, va="top")

        # ── Metrics table ──────────────────────────────────────────────
        ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
        ax.axis("off")

        p = self.perf

        def fmt_pct(v):
            return f"{v:.2%}" if pd.notna(v) else "—"

        def fmt_x(v, decimals=2):
            return f"{v:.{decimals}f}x" if pd.notna(v) else "—"

        def fmt_f(v, decimals=2):
            return f"{v:.{decimals}f}" if pd.notna(v) else "—"

        # Define all rows: (label, strategy_value, benchmark_value)
        rows = [
            ("RETURN METRICS", "", ""),
            ("CAGR (annualized)",          fmt_pct(p.get("cagr")),             fmt_pct(p.get("benchmark_cagr"))),
            ("Total Return",               fmt_pct(p.get("total_return")),      fmt_pct(p.get("benchmark_total"))),
            ("Annualized Volatility",      fmt_pct(p.get("volatility")),        fmt_pct(p.get("benchmark_vol"))),
            ("", "", ""),
            ("RISK-ADJUSTED METRICS", "", ""),
            ("Sharpe Ratio",               fmt_f(p.get("sharpe")),              fmt_f(p.get("benchmark_sharpe"))),
            ("Sortino Ratio",              fmt_f(p.get("sortino")),             "—"),
            ("Calmar Ratio",               fmt_f(p.get("calmar")),              "—"),
            ("", "", ""),
            ("DRAWDOWN & TAIL RISK", "", ""),
            ("Maximum Drawdown",           fmt_pct(p.get("max_drawdown")),      fmt_pct(p.get("benchmark_max_dd"))),
            ("VaR (95%, daily)",           fmt_pct(self.risk.get("var_95")),    fmt_pct(self.risk.get("benchmark_var_95"))),
            ("CVaR (95%, daily)",          fmt_pct(self.risk.get("cvar_95")),   fmt_pct(self.risk.get("benchmark_cvar_95"))),
            ("", "", ""),
            ("RELATIVE METRICS", "", ""),
            ("Information Ratio",          fmt_f(p.get("information_ratio")),   "—"),
            ("Alpha (annual)",             fmt_pct(p.get("alpha")),             "—"),
            ("Beta",                       fmt_f(p.get("beta")),                "1.00"),
            ("Monthly Win Rate",           fmt_pct(p.get("monthly_win_rate")),  "—"),
        ]

        col_labels = ["Metric", "Strategy", "Benchmark (SPY)"]
        col_x      = [0.02, 0.60, 0.80]

        # Header row
        for x, label in zip(col_x, col_labels):
            ax.text(
                x, 0.97, label,
                fontsize=11, fontweight="bold", color=DARK,
                transform=ax.transAxes, va="top",
            )

        # Note: transform= is not allowed in axhline in matplotlib 3.10+
        # We draw the rule manually using Line2D instead
        rule = plt.Line2D(
            [0, 1], [0.965, 0.965],
            transform=ax.transAxes,
            color=MID_GRAY, linewidth=1.5,
        )
        ax.add_artist(rule)

        # Data rows
        y       = 0.93
        row_h   = 0.041
        for i, (label, strat_val, bench_val) in enumerate(rows):
            bg_color = LIGHT_GRAY if i % 2 == 0 else "white"

            # Section headers
            if label.isupper() and strat_val == "":
                ax.text(col_x[0], y, label, fontsize=9,
                        fontweight="bold", color=STRATEGY_COLOR,
                        transform=ax.transAxes, va="top")
                y -= row_h * 0.8
                continue

            if label == "":
                y -= row_h * 0.3
                continue

            # Draw light row background
            ax.axhspan(y - row_h * 0.75, y + row_h * 0.1,
                       xmin=0, xmax=1, color=bg_color,
                       transform=ax.transAxes, alpha=0.5)

            ax.text(col_x[0], y, label, fontsize=10,
                    color=DARK, transform=ax.transAxes, va="top")
            ax.text(col_x[1], y, strat_val, fontsize=10,
                    color=STRATEGY_COLOR, fontweight="bold",
                    transform=ax.transAxes, va="top")
            ax.text(col_x[2], y, bench_val, fontsize=10,
                    color=BENCHMARK_COLOR,
                    transform=ax.transAxes, va="top")
            y -= row_h

        return fig

    # ──────────────────────────────────────────────────────────────────────
    def _make_returns_page(self) -> plt.Figure:
        """Page 2 — cumulative returns and drawdown."""

        from analytics.risk import compute_rolling_drawdown

        fig = plt.figure(figsize=(11, 8.5))
        self._header(fig, "Cumulative Returns & Drawdown Analysis")

        # ── Cumulative returns (top 60%) ───────────────────────────────
        ax1 = fig.add_axes([0.07, 0.45, 0.90, 0.46])

        # Use each series independently to avoid empty aligned result
        strat = self.strat.dropna()
        bench = self.bench.dropna()

        if len(strat) == 0 or len(bench) == 0:
            ax1.text(0.5, 0.5, "Insufficient return data",
                     ha="center", va="center", transform=ax1.transAxes)
        else:
            sw = (1 + strat).cumprod() * self.cap / 1e6
            bw = (1 + bench).cumprod() * self.cap / 1e6

            ax1.plot(sw.index, sw, color=STRATEGY_COLOR, lw=2,
                     label=f"Strategy  ${sw.iloc[-1]:.2f}M")
            ax1.plot(bw.index, bw, color=BENCHMARK_COLOR, lw=1.5, ls="--",
                     label=f"SPY       ${bw.iloc[-1]:.2f}M")

            common = sw.index.intersection(bw.index)
            if len(common) > 0:
                sw_c = sw.reindex(common)
                bw_c = bw.reindex(common)
                ax1.fill_between(common, sw_c, bw_c,
                                 where=sw_c >= bw_c, alpha=0.08, color=STRATEGY_COLOR)
                ax1.fill_between(common, sw_c, bw_c,
                                 where=sw_c < bw_c, alpha=0.08, color=LOSS_COLOR)

            ax1.set_title("Growth of $1,000,000", fontsize=12, fontweight="bold")
            ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
            ax1.legend(loc="upper left", framealpha=0.9)
            ax1.grid(True, alpha=0.3)

        # ── Drawdown (bottom 35%) ──────────────────────────────────────
        ax2 = fig.add_axes([0.07, 0.07, 0.90, 0.32])
        if len(strat) > 0:
            sd  = compute_rolling_drawdown(strat) * 100
            bd  = compute_rolling_drawdown(bench)  * 100 if len(bench) > 0 else pd.Series(dtype=float)

            ax2.fill_between(sd.index, sd, 0, alpha=0.4, color=STRATEGY_COLOR)
            if len(bd) > 0:
                ax2.fill_between(bd.index, bd, 0, alpha=0.2, color=BENCHMARK_COLOR)
                ax2.plot(bd.index, bd, color=BENCHMARK_COLOR, lw=0.8, ls="--")
            ax2.plot(sd.index, sd, color=STRATEGY_COLOR, lw=0.8)

        ax2.set_title("Drawdown from Peak", fontsize=12, fontweight="bold")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True, alpha=0.3)

        return fig

    # ──────────────────────────────────────────────────────────────────────
    def _make_rolling_page(self) -> plt.Figure:
        """Page 3 — rolling Sharpe + annual returns."""

        from analytics.risk import compute_rolling_sharpe

        fig = plt.figure(figsize=(11, 8.5))
        self._header(fig, "Rolling Performance & Annual Returns")

        # ── Rolling Sharpe (top 45%) ───────────────────────────────────
        ax1 = fig.add_axes([0.07, 0.52, 0.90, 0.38])
        rs  = compute_rolling_sharpe(self.strat, 252)

        ax1.axhline(0, color="#9CA3AF", lw=1, ls="--")
        ax1.axhline(1, color=PROFIT_COLOR, lw=0.8, ls=":", alpha=0.6)
        ax1.fill_between(rs.index, rs, 0, where=rs >= 0,
                         alpha=0.25, color=PROFIT_COLOR)
        ax1.fill_between(rs.index, rs, 0, where=rs < 0,
                         alpha=0.25, color=LOSS_COLOR)
        ax1.plot(rs.index, rs, color=STRATEGY_COLOR, lw=1.5)
        ax1.set_title("Rolling 12-Month Sharpe Ratio", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.grid(True, alpha=0.3)

        # ── Annual returns (bottom 40%) ────────────────────────────────
        ax2 = fig.add_axes([0.07, 0.07, 0.90, 0.38])

        strat_clean = self.strat.dropna()
        bench_clean = self.bench.dropna()

        # Ensure DatetimeIndex before resampling
        if not isinstance(strat_clean.index, pd.DatetimeIndex):
            strat_clean.index = pd.to_datetime(strat_clean.index)
        if not isinstance(bench_clean.index, pd.DatetimeIndex):
            bench_clean.index = pd.to_datetime(bench_clean.index)

        sa = (1 + strat_clean).resample("YE").prod() - 1
        ba = (1 + bench_clean).resample("YE").prod() - 1
        al = pd.concat([sa, ba], axis=1).dropna()

        if len(al) > 0:
            years  = al.index.year.astype(str)
            sv     = al.iloc[:, 0].values * 100
            bv     = al.iloc[:, 1].values * 100
            x      = np.arange(len(years))
            w      = 0.38

            sc = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in sv]
            ax2.bar(x - w/2, sv, w, color=sc, alpha=0.85, label="Strategy")
            ax2.bar(x + w/2, bv, w, color=BENCHMARK_COLOR, alpha=0.55, label="SPY")
            ax2.axhline(0, color="#9CA3AF", lw=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax2.legend(loc="upper right", framealpha=0.9)

        ax2.set_title("Annual Returns — Strategy vs SPY", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        return fig

    # ──────────────────────────────────────────────────────────────────────
    def _make_detail_page(self) -> plt.Figure:
        """Page 4 — monthly heatmap and factor attribution."""

        from analytics.risk import compute_monthly_returns_table
        from matplotlib.colors import LinearSegmentedColormap

        fig = plt.figure(figsize=(11, 8.5))
        self._header(fig, "Monthly Returns & Factor Attribution")

        # ── Monthly heatmap (top 52%) ──────────────────────────────────
        ax1 = fig.add_axes([0.07, 0.50, 0.90, 0.42])
        ax1.axis("off")

        strat_dt = self.strat.dropna()
        if not isinstance(strat_dt.index, pd.DatetimeIndex):
            strat_dt.index = pd.to_datetime(strat_dt.index)
        table     = compute_monthly_returns_table(strat_dt)
        month_cols= [c for c in table.columns if c != "Full Year"]
        annual_col= table.get("Full Year")
        data      = table[month_cols].values * 100

        cmap = LinearSegmentedColormap.from_list(
            "rwg", [LOSS_COLOR, "white", PROFIT_COLOR], N=256
        )
        vmax = min(abs(np.nanpercentile(data, 97)), 10)

        # Draw cells manually
        n_years, n_months = data.shape
        cell_w = 0.065
        cell_h = 0.85 / max(n_years, 1)
        x0, y0 = 0.04, 0.95

        # Column headers
        for j, m in enumerate(month_cols):
            ax1.text(x0 + (j + 0.5) * cell_w, y0 + 0.03, m,
                     ha="center", va="bottom", fontsize=7.5,
                     fontweight="bold", color=DARK,
                     transform=ax1.transAxes)

        ax1.text(x0 + (n_months + 0.5) * cell_w, y0 + 0.03, "Year",
                 ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color=DARK, transform=ax1.transAxes)

        norm = plt.Normalize(-vmax, vmax)
        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        for i, year in enumerate(table.index):
            y_pos = y0 - (i + 0.5) * cell_h
            # Year label
            ax1.text(x0 - 0.01, y_pos, str(year),
                     ha="right", va="center", fontsize=7.5,
                     color=DARK, transform=ax1.transAxes)

            for j in range(n_months):
                val   = data[i, j]
                if np.isnan(val):
                    continue
                color = sm.to_rgba(val)
                rect  = plt.Rectangle(
                    (x0 + j * cell_w, y_pos - cell_h * 0.45),
                    cell_w * 0.92, cell_h * 0.85,
                    transform=ax1.transAxes,
                    color=color, zorder=1,
                )
                ax1.add_patch(rect)
                tc = "white" if abs(val) > vmax * 0.6 else DARK
                ax1.text(x0 + (j + 0.5) * cell_w, y_pos,
                         f"{val:.1f}",
                         ha="center", va="center", fontsize=6.5,
                         color=tc, fontweight="bold",
                         transform=ax1.transAxes, zorder=2)

            # Annual return
            if annual_col is not None and year in annual_col.index:
                ann = annual_col[year] * 100
                c   = PROFIT_COLOR if ann >= 0 else LOSS_COLOR
                ax1.text(x0 + (n_months + 0.5) * cell_w, y_pos,
                         f"{ann:+.1f}%",
                         ha="center", va="center", fontsize=7,
                         color=c, fontweight="bold",
                         transform=ax1.transAxes)

        ax1.set_title("Monthly Returns (%)", fontsize=12,
                      fontweight="bold", pad=16)

        # ── Factor attribution (bottom 36%) ────────────────────────────
        ax2 = fig.add_axes([0.07, 0.07, 0.90, 0.35])

        if self.attr_df is not None and not self.attr_df.empty:
            ann_ret = {}
            for col in self.attr_df.columns:
                m = self.attr_df[col].dropna().mean()
                ann_ret[col] = (1 + m) ** 12 - 1

            labels = [f.replace("_", " ").title() for f in ann_ret]
            values = [v * 100 for v in ann_ret.values()]
            colors = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in values]

            bars = ax2.bar(labels, values, color=colors, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, values):
                y_off = 0.15 if val >= 0 else -0.35
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    f"{val:+.1f}%",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9.5, fontweight="bold",
                    color=PROFIT_COLOR if val >= 0 else LOSS_COLOR,
                )
        else:
            ax2.text(0.5, 0.5, "Factor attribution data not available",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=11)

        ax2.axhline(0, color="#9CA3AF", lw=0.8)
        ax2.set_title("Factor Attribution — Annualized Return by Factor",
                      fontsize=12, fontweight="bold")
        ax2.set_ylabel("Annualized Return (%)")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True, alpha=0.3)

        return fig
