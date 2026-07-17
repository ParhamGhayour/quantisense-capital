
"""
Quantisense Capital
ATR Volatility Module
"""

import pandas as pd


def _get_column(df, name):
    """
    Handle yfinance MultiIndex columns.
    """

    if isinstance(df.columns, pd.MultiIndex):
        return df[name].iloc[:, 0]

    return df[name]


def add_true_range(df):
    """
    Calculate True Range.
    """

    df = df.copy()

    high = _get_column(df, "High")
    low = _get_column(df, "Low")
    close = _get_column(df, "Close")

    high_low = high - low

    high_close = (
        high - close.shift(1)
    ).abs()

    low_close = (
        low - close.shift(1)
    ).abs()

    df["True_Range"] = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return df


def add_atr(df, window=14):
    """
    Calculate Average True Range.
    """

    df = df.copy()

    close = _get_column(df, "Close")

    df["ATR"] = (
        df["True_Range"]
        .rolling(window)
        .mean()
    )

    df["ATR_Percent"] = (
        df["ATR"] / close
    )

    return df
