"""
backtest/rebalance.py
---------------------
This file handles the mechanical details of rebalancing a portfolio.

"Rebalancing" means: at the end of each month, we look at our factor
scores, decide which 50 stocks we want to hold next month, and make
the necessary trades to get there.

In the real world, trading is not free. Every time you buy or sell a
stock you pay a small cost in the "bid-ask spread" — the difference
between what buyers will pay and what sellers want. We model this as
10 basis points (0.10%) per trade, one-way.

Example:
  You have $1,000,000 in the portfolio.
  You sell $50,000 of stock A and buy $50,000 of stock B.
  Transaction cost = $50,000 × 0.10% = $50 on the sell
                   + $50,000 × 0.10% = $50 on the buy
                   = $100 total cost for that pair of trades.

Over 15 years of monthly rebalancing, these small costs add up and
meaningfully drag on performance — which is why we model them carefully.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_rebalance_trades(
    current_weights:  dict[str, float],
    target_weights:   dict[str, float],
) -> dict[str, float]:
    """
    Figures out how much of each stock to buy or sell.

    Parameters
    ----------
    current_weights : {ticker: weight} — what we currently hold
                      Empty dict on the first rebalance (we hold nothing).
    target_weights  : {ticker: weight} — what we want to hold next month

    Returns
    -------
    dict {ticker: trade_weight}
        Positive = buy this much of the portfolio
        Negative = sell this much of the portfolio

    Example:
        current = {'AAPL': 0.05, 'MSFT': 0.05, 'GOOGL': 0.05}
        target  = {'AAPL': 0.04, 'MSFT': 0.05, 'TSLA': 0.06}
        trades  = {'AAPL': -0.01,  # sell some AAPL
                   'MSFT':  0.00,  # no change
                   'GOOGL': -0.05, # sell all GOOGL (exiting)
                   'TSLA':  +0.06} # buy TSLA (new position)
    """

    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    trades = {}

    for ticker in all_tickers:
        current = current_weights.get(ticker, 0.0)
        target  = target_weights.get(ticker,  0.0)
        trade   = target - current
        if abs(trade) > 1e-8:   # ignore rounding noise
            trades[ticker] = trade

    return trades


def apply_transaction_costs(
    trades:           dict[str, float],
    portfolio_value:  float,
    cost_bps:         int = 10,
) -> float:
    """
    Computes the total dollar cost of all trades at a rebalance.

    Parameters
    ----------
    trades          : output of compute_rebalance_trades()
    portfolio_value : current total portfolio value in dollars
    cost_bps        : transaction cost in basis points (1 bps = 0.01%)
                      Default 10 bps from strategy.yaml

    Returns
    -------
    float : total transaction cost in dollars

    Notes
    -----
    We only charge on the trades that actually move money (non-zero trades).
    We charge on the gross trade size (absolute value) since both buys
    and sells incur costs.
    """

    # Convert basis points to a decimal fraction
    # 10 bps = 10 / 10,000 = 0.0010 = 0.10%
    cost_rate = cost_bps / 10_000

    # Total traded = sum of absolute weight changes × portfolio value
    total_traded = sum(abs(v) for v in trades.values()) * portfolio_value

    total_cost = total_traded * cost_rate

    if total_cost > 0:
        cost_pct = total_cost / portfolio_value
        logger.debug(
            f"    Transaction costs: ${total_cost:,.0f} "
            f"({cost_pct:.3%} of portfolio)"
        )

    return total_cost


class RebalanceRecord:
    """
    A simple data container that stores everything about one rebalance event.

    Think of it like a snapshot in time capturing:
      - When did we rebalance?
      - What stocks did we buy and sell?
      - What did the trades cost us?
      - What does the portfolio look like now?
    """

    def __init__(
        self,
        date:             pd.Timestamp,
        prev_weights:     dict,
        new_weights:      dict,
        trades:           dict,
        transaction_cost: float,
        portfolio_value:  float,
    ):
        self.date             = date
        self.prev_weights     = prev_weights
        self.new_weights      = new_weights
        self.trades           = trades
        self.transaction_cost = transaction_cost
        self.portfolio_value  = portfolio_value

        # Derived stats
        self.n_holdings   = len([w for w in new_weights.values() if w > 0])
        self.n_buys       = len([t for t in trades.values() if t > 0])
        self.n_sells      = len([t for t in trades.values() if t < 0])
        self.turnover     = sum(abs(t) for t in trades.values()) / 2  # two-sided → one-sided

    def __repr__(self):
        return (
            f"RebalanceRecord({self.date.date()} | "
            f"{self.n_holdings} holdings | "
            f"{self.n_buys} buys, {self.n_sells} sells | "
            f"turnover={self.turnover:.1%} | "
            f"cost=${self.transaction_cost:,.0f})"
        )
