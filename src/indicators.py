
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
