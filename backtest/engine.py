"""
backtest/engine.py
------------------
Simulates monthly rebalancing of a factor-based equity portfolio.
Walks through time month by month, recomputes factor scores at each
rebalance date using only data available at that point (no lookahead),
and tracks daily portfolio value throughout.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:

    def __init__(self, config, prices, fundamentals):
        self.config       = config
        self.prices       = prices
        self.fundamentals = fundamentals
        self.start_date   = pd.Timestamp(config["start_date"])
        self.end_date     = pd.Timestamp(config["end_date"])
        self.portfolio_size  = config["portfolio_size"]
        self.weighting       = config["weighting"]
        self.rebalance_freq  = config["rebalance_freq"]
        self.cost_bps        = config["transaction_cost_bps"]
        self.benchmark       = config["benchmark"]
        self.initial_capital = 1_000_000

    def run(self) -> dict:

        logger.info("=" * 60)
        logger.info("  BACKTEST ENGINE STARTING")
        logger.info("=" * 60)
        logger.info(f"  Period    : {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"  Universe  : {self.prices.shape[1]} stocks")
        logger.info(f"  Rebalance : {self.rebalance_freq}")
        logger.info(f"  Port size : {self.portfolio_size} stocks")
        logger.info(f"  Weighting : {self.weighting}")
        logger.info(f"  Cost      : {self.cost_bps} bps/trade")

        # Trim prices to backtest window with DatetimeIndex guaranteed
        prices = self.prices.copy()
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        prices = prices.loc[self.start_date:self.end_date]
        logger.info(f"  Price data: {prices.shape[0]} trading days x {prices.shape[1]} tickers")

        # Separate benchmark from stock prices
        # We keep SPY in prices for beta calculation but exclude from portfolio
        benchmark_ticker = self.benchmark
        stock_prices = prices.drop(columns=[benchmark_ticker], errors="ignore")

        rebalance_dates = self._get_rebalance_dates(prices)
        logger.info(f"  Rebalance dates: {len(rebalance_dates)} periods")

        from backtest.rebalance import (
            compute_rebalance_trades,
            apply_transaction_costs,
            RebalanceRecord,
        )

        # These track the simulation state
        portfolio_dates  = []   # list of dates
        portfolio_vals   = []   # list of dollar values
        holdings_history = {}
        rebalance_log    = []
        current_weights  = {}
        portfolio_value  = self.initial_capital

        for i, rebal_date in enumerate(rebalance_dates):

            logger.info(f"\n  [{i+1}/{len(rebalance_dates)}] Rebalancing on {rebal_date.date()}...")

            # Only use data available UP TO this date (no lookahead)
            prices_to_date = prices.loc[:rebal_date]
            new_weights    = self._compute_weights_at_date(prices_to_date, stock_prices.loc[:rebal_date])

            if not new_weights:
                logger.warning("    Could not compute weights — skipping rebalance.")
                continue

            trades = compute_rebalance_trades(current_weights, new_weights)
            cost   = apply_transaction_costs(trades, portfolio_value, self.cost_bps)
            portfolio_value -= cost

            record = RebalanceRecord(
                date             = rebal_date,
                prev_weights     = current_weights.copy(),
                new_weights      = new_weights.copy(),
                trades           = trades,
                transaction_cost = cost,
                portfolio_value  = portfolio_value,
            )
            rebalance_log.append(record)
            logger.info(f"    {record}")

            current_weights              = new_weights.copy()
            holdings_history[rebal_date] = new_weights.copy()

            # Simulate daily returns until next rebalance
            next_rebal    = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else self.end_date
            # Slice from the day AFTER rebal_date so we get the full
            # holding period. Using rebal_date as start gives only 1 row
            # (just the rebalance date itself) which pct_change drops entirely.
            period_start  = prices.index[prices.index.get_loc(rebal_date) + 1]                             if rebal_date in prices.index and                                prices.index.get_loc(rebal_date) + 1 < len(prices.index)                             else rebal_date
            period_prices = stock_prices.loc[period_start:next_rebal]

            if len(period_prices) < 2:
                continue

            # Only compute returns for stocks we actually hold.
            # Using dropna() on the full 500-stock DataFrame drops entire rows
            # because many stocks have NaN prices in early years (not yet public).
            # Fix: slice to held stocks only, then fill any remaining NaN with 0.
            held_tickers   = list(current_weights.keys())
            held_available = [t for t in held_tickers if t in period_prices.columns]

            if not held_available:
                continue

            held_prices    = period_prices[held_available]
            period_returns = held_prices.pct_change().iloc[1:]  # drop first NaN row only

            weight_series  = pd.Series(current_weights)

            for date, daily_row in period_returns.iterrows():
                common = weight_series.index.intersection(daily_row.index)
                if common.empty:
                    portfolio_dates.append(date)
                    portfolio_vals.append(portfolio_value)
                    continue
                w        = weight_series[common]
                r        = daily_row[common].fillna(0)
                port_ret = float(w.dot(r))
                portfolio_value *= (1 + port_ret)
                portfolio_dates.append(date)
                portfolio_vals.append(portfolio_value)

        # Build output series with explicit DatetimeIndex
        logger.info("\n  Building result series...")

        dt_index = pd.DatetimeIndex(portfolio_dates)
        port_value_series = pd.Series(portfolio_vals, index=dt_index)
        port_value_series = port_value_series.sort_index()
        port_value_series = port_value_series[~port_value_series.index.duplicated(keep="last")]
        port_value_series.name = "portfolio_value"

        portfolio_returns      = port_value_series.pct_change().dropna()
        portfolio_returns.name = "strategy"

        logger.info(f"  Portfolio returns: {len(portfolio_returns)} days, "
                    f"from {portfolio_returns.index[0].date() if len(portfolio_returns) > 0 else 'N/A'} "
                    f"to {portfolio_returns.index[-1].date() if len(portfolio_returns) > 0 else 'N/A'}")

        # Benchmark returns
        benchmark_returns = self._get_benchmark_returns(prices)

        holdings_df = pd.DataFrame(holdings_history).T
        holdings_df.index = pd.DatetimeIndex(holdings_df.index)
        holdings_df.index.name = "rebalance_date"

        # Summary stats
        if len(portfolio_returns) > 0:
            total_return = (port_value_series.iloc[-1] / self.initial_capital) - 1
            years        = len(portfolio_returns) / 252
            cagr         = (1 + total_return) ** (1 / max(years, 0.01)) - 1
            total_costs  = sum(r.transaction_cost for r in rebalance_log)
            avg_turnover = np.mean([r.turnover for r in rebalance_log]) if rebalance_log else 0

            logger.info("\n" + "=" * 60)
            logger.info("  BACKTEST COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Starting capital : ${self.initial_capital:>15,.0f}")
            logger.info(f"  Ending value     : ${port_value_series.iloc[-1]:>15,.0f}")
            logger.info(f"  Total return     : {total_return:>14.1%}")
            logger.info(f"  CAGR (approx)    : {cagr:>14.1%}")
            logger.info(f"  Total costs paid : ${total_costs:>15,.0f}")
            logger.info(f"  Avg monthly turn : {avg_turnover:>14.1%}")
            logger.info(f"  Rebalances done  : {len(rebalance_log):>14}")
            logger.info("=" * 60)

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
            "portfolio_value":   port_value_series,
            "holdings_history":  holdings_df,
            "rebalance_log":     rebalance_log,
            "initial_capital":   self.initial_capital,
        }

    def _get_rebalance_dates(self, prices: pd.DataFrame) -> list:
        freq_map = {"monthly": "ME", "quarterly": "QE"}
        freq     = freq_map.get(self.rebalance_freq, "ME")
        dates    = prices.resample(freq).last().index.tolist()
        return [d for d in dates if self.start_date <= d <= self.end_date]

    def _compute_weights_at_date(self, prices_to_date: pd.DataFrame,
                                  stock_prices_to_date: pd.DataFrame) -> dict:
        try:
            from factors.composite import CompositeFactorBuilder
            from portfolio.construction import select_portfolio
            from portfolio.weighting import get_weights

            # Use full price data (including SPY) for factor computation
            # but stock-only prices for weighting
            builder = CompositeFactorBuilder(
                self.config, prices_to_date, self.fundamentals
            )
            composite_scores = builder.build()
            selected         = select_portfolio(composite_scores, self.portfolio_size)

            if not selected:
                return {}

            return get_weights(
                tickers   = selected,
                weighting = self.weighting,
                prices    = stock_prices_to_date,
            )
        except Exception as e:
            logger.error(f"    Weight computation failed: {e}")
            return {}

    def _get_benchmark_returns(self, prices: pd.DataFrame) -> pd.Series:
        if self.benchmark in prices.columns:
            bench = prices[self.benchmark].pct_change().dropna()
            if not isinstance(bench.index, pd.DatetimeIndex):
                bench.index = pd.to_datetime(bench.index)
            bench.name = "benchmark"
            logger.info(f"  Benchmark ({self.benchmark}) returns: {len(bench)} days")
            return bench
        logger.warning(f"  Benchmark {self.benchmark} not in price data.")
        return pd.Series(dtype=float, name="benchmark",
                         index=pd.DatetimeIndex([]))
