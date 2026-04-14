"""
tests/test_pipeline.py
-----------------------
A fast "smoke test" that runs the entire pipeline on a small dataset.

What is a smoke test?
  The name comes from electronics — if you turn on a circuit and it
  starts smoking, something is very wrong. A software smoke test is the
  same idea: run the whole system quickly to check nothing catastrophically
  breaks before committing to the full long run.

This test uses:
  - 10 well-known stocks instead of 500
  - 2 years of history instead of 15
  - A portfolio of 5 stocks instead of 50

It completes in under 5 minutes and tells you immediately if there are
any import errors, missing files, or logic crashes.

HOW TO RUN:
    python tests/test_pipeline.py

WHAT TO EXPECT:
  If everything is working you will see:
    ✓ Data ingestion passed
    ✓ Factor construction passed
    ✓ Portfolio construction passed
    ✓ Backtest engine passed
    ✓ Performance analytics passed
    ✓ Reporting passed
    ══════════════════════════
    ALL TESTS PASSED — ready for full run!
"""

import sys
import os
import logging
import traceback

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,    # suppress INFO logs during test for clean output
    format="%(levelname)s | %(name)s — %(message)s"
)

# ── Test configuration (small and fast) ───────────────────────────────────────
TEST_CONFIG = {
    "universe":             "sp500",
    "start_date":           "2022-01-01",
    "end_date":             "2023-12-31",
    "benchmark":            "SPY",
    "portfolio_size":       5,
    "weighting":            "equal_weight",
    "rebalance_freq":       "monthly",
    "transaction_cost_bps": 10,
    "factors": {
        "value":          {"enabled": True,  "weight": 0.25},
        "momentum":       {"enabled": True,  "weight": 0.25},
        "quality":        {"enabled": True,  "weight": 0.20},
        "low_volatility": {"enabled": True,  "weight": 0.15},
        "size":           {"enabled": True,  "weight": 0.15},
    },
    "zscore": {
        "winsorize_pct":  0.01,
        "cross_sectional": True,
    },
    "output": {
        "log_level":      "WARNING",
        "output_dir":     "reporting/output",
    }
}

# 10 large, liquid, well-known stocks that are unlikely to have data gaps
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ",
                "XOM", "UNH", "V", "PG", "SPY"]


def section(title: str):
    print(f"\n  Testing: {title}...")


def passed(title: str):
    print(f"  ✓ {title} passed")


def failed(title: str, error: Exception):
    print(f"  ✗ {title} FAILED")
    print(f"    Error: {type(error).__name__}: {error}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════

def test_data_ingestion():
    section("Data ingestion")
    from data.ingest import download_prices, download_fundamentals

    prices = download_prices(
        TEST_TICKERS,
        start_date    = TEST_CONFIG["start_date"],
        end_date      = TEST_CONFIG["end_date"],
        force_refresh = True,
    )
    assert prices.shape[0] > 100,   f"Expected 100+ rows, got {prices.shape[0]}"
    assert prices.shape[1] > 5,     f"Expected 5+ tickers, got {prices.shape[1]}"
    assert not prices.empty,         "Price data is empty"

    fund = download_fundamentals(TEST_TICKERS, force_refresh=True)
    assert len(fund) > 0,            "Fundamentals DataFrame is empty"
    assert "pb_ratio" in fund.columns, "Missing pb_ratio column"

    passed("Data ingestion")
    return prices, fund


def test_factor_construction(prices, fund):
    section("Factor construction")
    from factors.value    import compute_value_scores
    from factors.momentum import compute_momentum_scores
    from factors.quality  import compute_quality_scores
    from factors.low_vol  import compute_low_vol_scores
    from factors.size     import compute_size_scores
    from factors.composite import CompositeFactorBuilder

    # Test each factor individually
    val = compute_value_scores(fund)
    assert len(val) > 0, "Value scores empty"

    mom = compute_momentum_scores(prices)
    assert len(mom) > 0, "Momentum scores empty"

    qual = compute_quality_scores(fund)
    assert len(qual) > 0, "Quality scores empty"

    lv = compute_low_vol_scores(prices)
    assert len(lv) > 0, "Low-vol scores empty"

    sz = compute_size_scores(fund)
    assert len(sz) > 0, "Size scores empty"

    # Test composite builder
    builder   = CompositeFactorBuilder(TEST_CONFIG, prices, fund)
    composite = builder.build()
    assert len(composite) > 0,      "Composite scores empty"
    assert composite.notna().sum() > 0, "All composite scores are NaN"

    passed("Factor construction")
    return composite


def test_portfolio_construction(composite, prices):
    section("Portfolio construction")
    from portfolio.construction import select_portfolio
    from portfolio.weighting    import get_weights, equal_weight, risk_parity_weight

    selected = select_portfolio(composite, portfolio_size=TEST_CONFIG["portfolio_size"])
    assert len(selected) > 0,             "No stocks selected"
    assert len(selected) <= TEST_CONFIG["portfolio_size"], "Too many stocks selected"

    # Equal weight
    ew = equal_weight(selected)
    assert abs(sum(ew.values()) - 1.0) < 1e-6, "Equal weights don't sum to 1.0"

    # Risk parity
    rp = risk_parity_weight(selected, prices)
    assert abs(sum(rp.values()) - 1.0) < 1e-6, "Risk parity weights don't sum to 1.0"

    # Dispatcher
    w = get_weights(selected, "equal_weight", prices)
    assert len(w) == len(selected), "Weight count mismatch"

    passed("Portfolio construction")
    return selected, ew


def test_backtest_engine(prices, fund):
    section("Backtest engine")
    from backtest.engine import BacktestEngine

    engine  = BacktestEngine(TEST_CONFIG, prices, fund)
    results = engine.run()

    assert "portfolio_returns" in results,  "Missing portfolio_returns"
    assert "benchmark_returns" in results,  "Missing benchmark_returns"
    assert "portfolio_value"   in results,  "Missing portfolio_value"
    assert "rebalance_log"     in results,  "Missing rebalance_log"

    port_ret = results["portfolio_returns"]
    assert len(port_ret) > 0,               "Portfolio returns are empty"
    assert port_ret.notna().sum() > 0,      "All portfolio returns are NaN"

    bench_ret = results["benchmark_returns"]
    assert len(bench_ret) > 0,              "Benchmark returns are empty"

    final_value = results["portfolio_value"].iloc[-1]
    assert final_value > 0,                 "Portfolio value went to zero or negative"
    assert len(results["rebalance_log"]) > 0, "No rebalances were recorded"

    passed("Backtest engine")
    return results


def test_analytics(results):
    section("Performance analytics")
    from analytics.performance import compute_all_metrics
    from analytics.risk        import compute_all_risk_metrics

    metrics = compute_all_metrics(
        results["portfolio_returns"],
        results["benchmark_returns"],
    )
    assert "cagr"       in metrics, "Missing CAGR"
    assert "sharpe"     in metrics, "Missing Sharpe"
    assert "max_drawdown" in metrics, "Missing max drawdown"
    assert metrics["cagr"] is not None, "CAGR is None"

    risk = compute_all_risk_metrics(
        results["portfolio_returns"],
        results["benchmark_returns"],
    )
    assert "var_95"           in risk, "Missing VaR 95%"
    assert "drawdown_series"  in risk, "Missing drawdown series"
    assert "rolling_sharpe_252" in risk, "Missing rolling Sharpe"

    passed("Performance analytics")
    return metrics, risk


def test_reporting(results, metrics, risk):
    section("Reporting / tearsheet")
    import pandas as pd
    from reporting.tearsheet import TearsheetGenerator
    from pathlib import Path

    # Pass an empty attribution DataFrame — attribution is slow for the smoke test
    empty_attr = pd.DataFrame()

    gen  = TearsheetGenerator(TEST_CONFIG, metrics, risk, empty_attr, results)
    path = gen.generate()

    assert Path(path).exists(), f"Tearsheet PDF not found at {path}"
    assert Path(path).stat().st_size > 10_000, "Tearsheet PDF is suspiciously small"

    passed("Reporting")
    return path


# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("\n" + "═" * 50)
    print("  SMOKE TEST — Multi-Factor Backtest Pipeline")
    print("  Using 10 tickers, 2022-2023 (fast mode)")
    print("═" * 50)

    all_passed = True

    try:
        prices, fund = test_data_ingestion()
    except Exception as e:
        failed("Data ingestion", e)
        print("\n  Cannot continue without data. Fix data ingestion first.")
        return False

    try:
        composite = test_factor_construction(prices, fund)
    except Exception as e:
        failed("Factor construction", e)
        all_passed = False
        composite  = None

    try:
        if composite is not None:
            test_portfolio_construction(composite, prices)
    except Exception as e:
        failed("Portfolio construction", e)
        all_passed = False

    try:
        results = test_backtest_engine(prices, fund)
    except Exception as e:
        failed("Backtest engine", e)
        print("\n  Cannot continue without backtest results.")
        return False

    try:
        metrics, risk = test_analytics(results)
    except Exception as e:
        failed("Performance analytics", e)
        all_passed = False
        metrics = {}
        risk    = {}

    try:
        tearsheet_path = test_reporting(results, metrics, risk)
        print(f"\n  Tearsheet saved to: {tearsheet_path}")
    except Exception as e:
        failed("Reporting", e)
        all_passed = False

    print("\n" + "═" * 50)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
        print("  You are ready to run the full backtest:")
        print("  → python main.py")
    else:
        print("  SOME TESTS FAILED ✗")
        print("  Review the errors above before running main.py")
    print("═" * 50 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
