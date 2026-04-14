"""
main.py — Entry point for the Multi-Factor Equity Strategy Backtest
Run: python main.py
"""

import yaml
import logging
from pathlib import Path


def load_config(path: str = "configs/strategy.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    config = load_config()
    setup_logging(config["output"]["log_level"])
    logger = logging.getLogger("main")

    logger.info("═" * 60)
    logger.info("  Multi-Factor Equity Strategy Backtest")
    logger.info("═" * 60)
    logger.info(f"  Universe  : {config['universe'].upper()}")
    logger.info(f"  Period    : {config['start_date']} → {config['end_date']}")
    logger.info(f"  Benchmark : {config['benchmark']}")
    logger.info(f"  Port. Size: {config['portfolio_size']} stocks")
    logger.info(f"  Weighting : {config['weighting']}")
    logger.info(f"  Rebalance : {config['rebalance_freq']}")
    logger.info("═" * 60)

    # ── Step 1: Data Ingestion ────────────────────────────────────
    logger.info("[1/5] Ingesting data...")
    from data.universe import get_sp500_tickers
    from data.ingest import DataIngestor

    tickers  = get_sp500_tickers()
    ingestor = DataIngestor(config)
    price_data, fundamental_data = ingestor.run(tickers)

    # ── Step 2: Factor Construction ───────────────────────────────
    logger.info("[2/5] Constructing factors...")
    from factors.composite import CompositeFactorBuilder
    builder = CompositeFactorBuilder(config, price_data, fundamental_data)
    composite_scores = builder.build()

    # ── Step 3: Portfolio Construction ────────────────────────────
    logger.info("[3/5] Building portfolios...")
    from portfolio.construction import select_portfolio
    from portfolio.weighting import get_weights

    selected_tickers = select_portfolio(
        composite_scores,
        portfolio_size=config["portfolio_size"]
    )
    weights = get_weights(
        tickers   = selected_tickers,
        weighting = config["weighting"],
        prices    = price_data,
    )
    logger.info(f"  Portfolio built: {len(weights)} stocks, weighting={config['weighting']}")

    # ── Step 4: Backtesting ───────────────────────────────────────
    logger.info("[4/5] Running backtest engine...")
    from backtest.engine import BacktestEngine
    engine  = BacktestEngine(config, price_data, fundamental_data)
    results = engine.run()

    # ── Step 5: Analytics & Reporting ────────────────────────────
    logger.info("[5/5] Running analytics and generating tearsheet...")
    import pandas as pd
    from analytics.performance import compute_all_metrics
    from analytics.risk import compute_all_risk_metrics
    from analytics.attribution import compute_factor_attribution
    from reporting.tearsheet import TearsheetGenerator

    # Ensure both return series have a proper DatetimeIndex
    port_ret  = results["portfolio_returns"].copy()
    bench_ret = results["benchmark_returns"].copy()
    if not isinstance(port_ret.index, pd.DatetimeIndex):
        port_ret.index  = pd.to_datetime(port_ret.index)
    if not isinstance(bench_ret.index, pd.DatetimeIndex):
        bench_ret.index = pd.to_datetime(bench_ret.index)
    results["portfolio_returns"] = port_ret
    results["benchmark_returns"] = bench_ret

    # DEBUG — print index info to terminal so we can diagnose
    import sys
    print("\n=== DEBUG: Return Series Info ===", file=sys.stderr)
    print(f"port_ret length     : {len(port_ret)}", file=sys.stderr)
    print(f"port_ret index type : {type(port_ret.index)}", file=sys.stderr)
    print(f"port_ret first 3    : {list(port_ret.index[:3])}", file=sys.stderr)
    print(f"port_ret last 3     : {list(port_ret.index[-3:])}", file=sys.stderr)
    print(f"bench_ret length    : {len(bench_ret)}", file=sys.stderr)
    print(f"bench_ret index type: {type(bench_ret.index)}", file=sys.stderr)
    print(f"bench_ret first 3   : {list(bench_ret.index[:3])}", file=sys.stderr)
    print("=================================\n", file=sys.stderr)

    perf_metrics = compute_all_metrics(port_ret, bench_ret)
    risk_metrics = compute_all_risk_metrics(port_ret, bench_ret)
    attr_df = compute_factor_attribution(
        price_data, fundamental_data, config, results["rebalance_log"]
    )

    TearsheetGenerator(
        config, perf_metrics, risk_metrics, attr_df, results
    ).generate()

    logger.info("✓ Backtest complete. Results saved to reporting/output/")


if __name__ == "__main__":
    main()
