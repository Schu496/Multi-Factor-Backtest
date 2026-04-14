"""
data/universe.py
----------------
This file is responsible for one job: giving us the list of S&P 500 stocks.

We fetch the list from Wikipedia (which keeps it reasonably up to date),
clean it up, and save it to a CSV file so we don't have to re-download
it every time we run the project.

Think of this as building our "shopping list" of stocks before we go
fetch all the price data.
"""

import pandas as pd
import requests
from pathlib import Path
import logging

# Set up a logger so we can see what this module is doing when it runs
logger = logging.getLogger(__name__)

# ── Where we save the cached file ─────────────────────────────────────────────
# Path() is just a clean way to describe a file location that works on
# both Mac and Windows without worrying about forward/back slashes.
UNIVERSE_CACHE_PATH = Path("data/universe/sp500_tickers.csv")


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    """
    Returns a list of S&P 500 ticker symbols (e.g. ['AAPL', 'MSFT', ...]).

    Parameters
    ----------
    force_refresh : bool
        If True, re-downloads the list even if a cached version exists.
        If False (default), uses the cached version if available.

    Returns
    -------
    list[str]
        A sorted list of ticker symbols.
    """

    # ── Step 1: Check if we already have a cached copy ────────────────────────
    # This avoids hammering Wikipedia every single time we run the code.
    if UNIVERSE_CACHE_PATH.exists() and not force_refresh:
        logger.info(f"Loading S&P 500 tickers from cache: {UNIVERSE_CACHE_PATH}")
        df = pd.read_csv(UNIVERSE_CACHE_PATH)
        tickers = df["ticker"].tolist()
        logger.info(f"  Loaded {len(tickers)} tickers from cache.")
        return tickers

    # ── Step 2: Download the list from Wikipedia ──────────────────────────────
    # Wikipedia has a well-maintained table of S&P 500 companies.
    # pandas can read HTML tables directly from a URL — very handy!
    logger.info("Downloading S&P 500 constituent list from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        df = tables[0]

    except Exception as e:
        logger.error(f"Failed to download S&P 500 list from Wikipedia: {e}")
        raise

    # ── Step 3: Clean up the data ─────────────────────────────────────────────
    # The Wikipedia table has a column called "Symbol" with the ticker.
    # Some tickers have dots (e.g. BRK.B) but yfinance uses dashes (BRK-B),
    # so we replace dots with dashes.
    tickers = (
        df["Symbol"]
        .str.strip()                    # remove any extra whitespace
        .str.replace(".", "-", regex=False)  # BRK.B → BRK-B for yfinance
        .sort_values()
        .tolist()
    )

    logger.info(f"  Downloaded {len(tickers)} tickers.")

    # ── Step 4: Save to CSV so we can reuse next time ─────────────────────────
    # Create the folder if it doesn't exist yet
    UNIVERSE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cache_df = pd.DataFrame({"ticker": tickers})
    cache_df.to_csv(UNIVERSE_CACHE_PATH, index=False)
    logger.info(f"  Saved ticker list to {UNIVERSE_CACHE_PATH}")

    return tickers


def get_sp500_metadata() -> pd.DataFrame:
    """
    Returns a DataFrame with extra info about each S&P 500 company:
    ticker, company name, sector, and sub-industry.

    This is useful later for sector-level analysis and attribution.
    """

    logger.info("Downloading S&P 500 metadata from Wikipedia...")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        tables = pd.read_html(url)
        df = tables[0]
    except Exception as e:
        logger.error(f"Failed to download metadata: {e}")
        raise

    # Keep only the columns we care about and rename them cleanly
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["ticker", "company_name", "sector", "sub_industry"]

    # Fix the BRK.B → BRK-B issue here too
    df["ticker"] = df["ticker"].str.strip().str.replace(".", "-", regex=False)

    logger.info(f"  Retrieved metadata for {len(df)} companies.")
    return df


# ── Quick test: run this file directly to see it work ─────────────────────────
# When you run `python data/universe.py` from your terminal, this block runs.
# When this file is imported by another module, this block is skipped.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n=== Fetching S&P 500 Universe ===")
    tickers = get_sp500_tickers(force_refresh=True)
    print(f"\nTotal tickers: {len(tickers)}")
    print(f"First 10: {tickers[:10]}")
    print(f"Last 10:  {tickers[-10:]}")

    print("\n=== Metadata sample ===")
    meta = get_sp500_metadata()
    print(meta.head(10).to_string(index=False))
