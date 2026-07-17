
"""
Quantisense Capital
Market Data Engine

Responsible for downloading and preparing financial market data.
"""

import yfinance as yf
import pandas as pd


def download_market_data(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Download OHLCV market data.

    Parameters:
        ticker: asset symbol (example: BTC-USD)
        start: start date YYYY-MM-DD
        end: end date YYYY-MM-DD

    Returns:
        DataFrame with OHLCV data
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False
    )

    return data
