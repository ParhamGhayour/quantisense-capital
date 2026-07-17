
"""
Quantisense Capital
Market Indicators Module
"""

import numpy as np


def add_returns(df):
    """
    Add simple and logarithmic returns.
    """

    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    return df


def add_volatility(df, window=7):
    """
    Add rolling volatility.
    """

    df = df.copy()

    df["Volatility"] = (
        df["Log_Return"]
        .rolling(window)
        .std()
    )

    return df


def add_ema_indicators(df):
    """
    Add exponential moving averages.
    """

    df = df.copy()

    for period in [9, 21, 50, 100, 250]:
        df[f"EMA_{period}"] = (
            df["Close"]
            .ewm(span=period, adjust=False)
            .mean()
        )

    return df
