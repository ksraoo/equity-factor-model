"""
Step 1: Data pipeline for the cross-sectional factor model project.
Pulls daily price data for a universe of tickers, computes forward returns,
and saves a clean panel dataset for factor construction.

Run this locally, not in a restricted sandbox, since it needs live internet
access to Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ---- CONFIG ----
START_DATE = "2019-01-01"
END_DATE = "2026-09-28"
OUTPUT_PATH = "data/price_panel.parquet"

# Start with a small, liquid test universe. Expand to full S&P 500 once
# this pipeline is validated end to end.
TEST_UNIVERSE = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "PFE", "V", "WMT",
    "HD", "UNH", "DIS", "BA", "CAT", "GE", "IBM", "INTC", "CSCO", "MCD"
]


def pull_price_data(tickers, start, end):
    """Download adjusted close prices for a list of tickers."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=False,
    )
    return data


def reshape_to_long(raw_data, tickers):
    """
    Reshape yfinance's multi-index wide format into a long panel:
    columns = [date, ticker, close, volume]
    This long format is much easier to work with for factor construction
    and merging with fundamentals later.
    """
    frames = []
    for ticker in tickers:
        try:
            df = raw_data[ticker][["Close", "Volume"]].copy()
            df["ticker"] = ticker
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            frames.append(df)
        except KeyError:
            print(f"Skipping {ticker}, no data returned")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["close"])
    return panel


def add_forward_returns(panel, horizon_days=21):
    """
    Add forward return column: the thing we'll eventually train the model
    to rank stocks by. horizon_days=21 is roughly one trading month.
    """
    panel = panel.sort_values(["ticker", "date"])
    panel["fwd_return"] = (
        panel.groupby("ticker")["close"]
        .transform(lambda x: x.shift(-horizon_days) / x - 1)
    )
    return panel


if __name__ == "__main__":
    print(f"Pulling data for {len(TEST_UNIVERSE)} tickers...")
    raw = pull_price_data(TEST_UNIVERSE, START_DATE, END_DATE)

    print("Reshaping to long panel format...")
    panel = reshape_to_long(raw, TEST_UNIVERSE)

    print("Adding forward returns...")
    panel = add_forward_returns(panel)

    print(panel.head(10))
    print(f"\nTotal rows: {len(panel)}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Tickers with data: {panel['ticker'].nunique()}")

    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
