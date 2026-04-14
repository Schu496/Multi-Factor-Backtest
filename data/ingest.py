"""
data/ingest.py
--------------
This file does the heavy lifting of actually downloading market data.

It handles two types of data:
  1. PRICE DATA   — daily adjusted closing prices for every stock (2010–2024)
  2. FUNDAMENTAL DATA — valuation metrics like P/E ratio, P/B ratio, market cap

All downloaded data gets saved to data/processed/ as CSV files so we
only need to download once. Subsequent runs load from disk instantly.

Why "adjusted" prices? Stock prices get adjusted for splits and dividends.
Without adjustment, a 2-for-1 stock split looks like a 50% price crash,
which would wreck our momentum calculations. Adjusted prices fix this.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm          # tqdm draws a progress bar in the terminal
import logging
import time

logger = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
PRICE_FILE    = PROCESSED_DIR / "prices_daily.csv"
FUND_FILE     = PROCESSED_DIR / "fundamentals.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_prices(
    tickers: list[str],
    start_date: str = "2010-01-01",
    end_date:   str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Downloads adjusted daily closing prices for all tickers.

    Returns a DataFrame where:
      - Each ROW is a date
      - Each COLUMN is a ticker symbol
      - Each CELL is the adjusted closing price on that date

    Example:
                AAPL    MSFT    GOOGL
    2010-01-04  27.43  28.05    NaN
    2010-01-05  27.83  27.98    NaN
    ...

    NaN means "no data" — either the stock wasn't public yet, or it
    was delisted. We handle that gracefully later.

    Parameters
    ----------
    tickers      : list of ticker symbols from universe.py
    start_date   : first date of data to download
    end_date     : last date of data to download
    force_refresh: re-download even if a local file already exists
    """

    # ── Check for cached version ───────────────────────────────────────────
    if PRICE_FILE.exists() and not force_refresh:
        logger.info(f"Loading prices from cache: {PRICE_FILE}")
        prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
        logger.info(f"  Loaded {prices.shape[1]} tickers × {prices.shape[0]} days.")
        return prices

    logger.info(f"Downloading prices for {len(tickers)} tickers ({start_date} → {end_date})...")
    logger.info("  This will take a few minutes. Grab a coffee ☕")

    # ── Create output directory if needed ─────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download in batches ────────────────────────────────────────────────
    # We download in groups of 50 tickers at a time.
    # Reason: asking for all 500 at once can cause timeouts.
    # Between batches we pause briefly to be polite to Yahoo Finance's servers.

    BATCH_SIZE  = 50
    all_prices  = []
    failed      = []

    # Split the ticker list into chunks of BATCH_SIZE
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for batch_num, batch in enumerate(tqdm(batches, desc="Downloading price batches")):
        try:
            # yf.download() is the main yfinance function.
            # auto_adjust=True gives us split/dividend-adjusted prices.
            # progress=False suppresses yfinance's own output (we use tqdm instead).
            raw = yf.download(
                tickers    = batch,
                start      = start_date,
                end        = end_date,
                auto_adjust= True,
                progress   = False,
            )

            # yfinance returns a multi-level column structure.
            # We only want the "Close" price column for each ticker.
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                # Single ticker edge case — wrap it
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            all_prices.append(close)

        except Exception as e:
            logger.warning(f"  Batch {batch_num} failed: {e}. Will retry individually.")
            failed.extend(batch)

        # Small pause between batches to avoid rate limiting
        time.sleep(1)

    # ── Retry any failed tickers one at a time ─────────────────────────────
    if failed:
        logger.info(f"  Retrying {len(failed)} failed tickers individually...")
        for ticker in tqdm(failed, desc="Retrying"):
            try:
                raw = yf.download(ticker, start=start_date, end=end_date,
                                  auto_adjust=True, progress=False)
                if not raw.empty:
                    close = raw[["Close"]].rename(columns={"Close": ticker})
                    all_prices.append(close)
            except Exception as e:
                logger.warning(f"    {ticker} permanently failed: {e}")
            time.sleep(0.5)

    # ── Combine all batches into one big DataFrame ─────────────────────────
    logger.info("  Combining all batches...")
    prices = pd.concat(all_prices, axis=1)

    # ── Data quality steps ─────────────────────────────────────────────────

    # 1. Sort the dates in chronological order
    prices = prices.sort_index()

    # 2. Remove any completely empty columns (stocks that had zero data)
    before = prices.shape[1]
    prices = prices.dropna(axis=1, how="all")
    after = prices.shape[1]
    if before != after:
        logger.info(f"  Dropped {before - after} tickers with no data at all.")

    # 3. Report data coverage
    coverage = prices.notna().mean()  # fraction of days each stock has data
    logger.info(f"  Median data coverage per ticker: {coverage.median():.1%}")
    logger.info(f"  Final shape: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # ── Save to disk ───────────────────────────────────────────────────────
    prices.to_csv(PRICE_FILE)
    logger.info(f"  Saved to {PRICE_FILE}")

    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_fundamentals(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Downloads fundamental (accounting) data for each ticker.

    "Fundamentals" are financial metrics derived from a company's
    balance sheet and income statement — things like:
      - P/E ratio  : Price divided by annual earnings per share
      - P/B ratio  : Price divided by book value per share
      - Market cap : Total value of all shares outstanding
      - ROE        : Return on equity (profitability measure)

    These are static snapshots (current values), not time series.
    In a production system you'd want historical quarterly data,
    but for this project we use current values as a starting point.

    Returns a DataFrame with one row per ticker.
    """

    # ── Check for cached version ───────────────────────────────────────────
    if FUND_FILE.exists() and not force_refresh:
        logger.info(f"Loading fundamentals from cache: {FUND_FILE}")
        fund = pd.read_csv(FUND_FILE, index_col=0)
        logger.info(f"  Loaded fundamentals for {len(fund)} tickers.")
        return fund

    logger.info(f"Downloading fundamentals for {len(tickers)} tickers...")
    logger.info("  This may take 5–10 minutes (one request per ticker).")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []  # We'll build up a list of dicts, then convert to DataFrame

    for ticker in tqdm(tickers, desc="Downloading fundamentals"):
        try:
            # yf.Ticker() creates an object for a single stock.
            # .info gives us a big dictionary of everything Yahoo knows about it.
            stock = yf.Ticker(ticker)
            info  = stock.info

            # Pull out the specific metrics we need for our factors.
            # The .get() method returns None if a field is missing
            # (rather than crashing with a KeyError).
            records.append({
                "ticker":           ticker,

                # ── Value factor inputs ──────────────────────────────────
                "pb_ratio":         info.get("priceToBook"),          # P/B ratio
                "pe_ratio":         info.get("trailingPE"),           # Trailing P/E
                "ev_ebitda":        info.get("enterpriseToEbitda"),   # EV/EBITDA

                # ── Quality factor inputs ────────────────────────────────
                "roe":              info.get("returnOnEquity"),        # Return on equity
                "gross_margin":     info.get("grossMargins"),         # Gross profit margin
                "debt_to_equity":   info.get("debtToEquity"),         # Leverage ratio

                # ── Size factor input ────────────────────────────────────
                "market_cap":       info.get("marketCap"),            # Market capitalisation

                # ── Metadata ─────────────────────────────────────────────
                "sector":           info.get("sector"),
                "industry":         info.get("industry"),
                "company_name":     info.get("longName"),
            })

        except Exception as e:
            # If a single ticker fails, log it and move on.
            # We don't want one bad ticker to crash the whole download.
            logger.warning(f"  {ticker}: failed ({e})")
            records.append({"ticker": ticker})   # add a blank row

        # Small pause to avoid overwhelming Yahoo Finance
        time.sleep(0.3)

    # ── Build DataFrame ────────────────────────────────────────────────────
    fund = pd.DataFrame(records).set_index("ticker")

    # ── Add log market cap (used directly in the size factor) ──────────────
    # We use the log (natural logarithm) of market cap because market caps
    # span a huge range ($1B to $3T). Log compresses that range so extreme
    # values don't dominate.
    fund["log_market_cap"] = np.log(fund["market_cap"].replace(0, np.nan))

    # ── Report coverage ────────────────────────────────────────────────────
    coverage = fund.notna().mean()
    logger.info("\n  Fundamental data coverage:")
    for col, pct in coverage.items():
        if col not in ["sector", "industry", "company_name"]:
            logger.info(f"    {col:<20}: {pct:.1%}")

    # ── Save to disk ───────────────────────────────────────────────────────
    fund.to_csv(FUND_FILE)
    logger.info(f"\n  Saved to {FUND_FILE}")

    return fund


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class DataIngestor:
    """
    A convenience class that bundles prices + fundamentals together.
    main.py will create one of these and call .run() to get everything.

    A "class" in Python is just a way to group related functions and
    data together under one name. Think of it like a machine:
    you feed it a config, call .run(), and it returns your data.
    """

    def __init__(self, config: dict):
        """
        config is the dictionary loaded from configs/strategy.yaml.
        We pull start/end dates and the universe name from it.
        """
        self.start_date = config["start_date"]
        self.end_date   = config["end_date"]

    def run(self, tickers: list[str], force_refresh: bool = False):
        """
        Downloads both prices and fundamentals.

        Returns
        -------
        prices : pd.DataFrame   — daily adjusted close prices (dates × tickers)
        fundamentals : pd.DataFrame — fundamental metrics (tickers × metrics)
        """
        logger.info("=== DataIngestor: starting data pipeline ===")

        # Always include SPY so the benchmark and beta calculations work.
        # SPY is the S&P 500 ETF used as our benchmark — it is not itself
        # a constituent of the index so it must be added explicitly.
        tickers_with_spy = list(set(tickers) | {"SPY"})

        prices       = download_prices(tickers_with_spy, self.start_date, self.end_date, force_refresh)
        fundamentals = download_fundamentals(tickers, force_refresh)

        logger.info("=== DataIngestor: complete ===")
        return prices, fundamentals


# ── Quick test: run this file directly ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Import our universe module to get the ticker list
    import sys
    sys.path.append(".")
    from data.universe import get_sp500_tickers

    # Use a small sample of 10 tickers to test quickly
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ",
                      "XOM", "BRK-B", "UNH", "V", "PG"]

    print("\n=== Testing price download (10 tickers) ===")
    prices = download_prices(sample_tickers, "2022-01-01", "2023-12-31", force_refresh=True)
    print(f"\nPrice data shape: {prices.shape}")
    print(prices.tail(3).round(2))

    print("\n=== Testing fundamentals download (10 tickers) ===")
    fund = download_fundamentals(sample_tickers, force_refresh=True)
    print(f"\nFundamentals shape: {fund.shape}")
    print(fund[["pb_ratio", "pe_ratio", "roe", "market_cap"]].round(2))
