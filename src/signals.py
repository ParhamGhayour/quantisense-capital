
"""
Quantisense Capital
Market Signal Engine

Combines:
- EMA trend structure
- ATR volatility regime
"""

import pandas as pd


def _get_column(df, name):
    """
    Handle yfinance MultiIndex columns.
    """

    if isinstance(df.columns, pd.MultiIndex):
        return df[name].iloc[:, 0]

    return df[name]


def add_ema_score(df):
    """
    Score EMA alignment.

    Returns:
    - EMA_Score (-100 to +100)
    - Trend
    """

    df = df.copy()

    ema9 = _get_column(df, "EMA_9")
    ema21 = _get_column(df, "EMA_21")
    ema50 = _get_column(df, "EMA_50")
    ema100 = _get_column(df, "EMA_100")
    ema250 = _get_column(df, "EMA_250")

    score = []

    for values in zip(
        ema9,
        ema21,
        ema50,
        ema100,
        ema250
    ):

        points = 0

        if values[0] > values[1]:
            points += 20
        else:
            points -= 20

        if values[1] > values[2]:
            points += 20
        else:
            points -= 20

        if values[2] > values[3]:
            points += 20
        else:
            points -= 20

        if values[3] > values[4]:
            points += 20
        else:
            points -= 20

        if values[0] > values[4]:
            points += 20
        else:
            points -= 20

        score.append(points)

    df["EMA_Score"] = score

    df["Trend"] = df["EMA_Score"].apply(
        lambda x:
        "Strong Bullish" if x >= 80 else
        "Bullish" if x >= 40 else
        "Neutral" if x > -40 else
        "Bearish" if x > -80 else
        "Strong Bearish"
    )

    return df


def add_volatility_regime(df):
    """
    Classify ATR volatility.
    """

    df = df.copy()

    atr_percent = df["ATR_Percent"]

    df["Volatility_Regime"] = atr_percent.apply(
        lambda x:
        "Low" if x < 0.01 else
        "Medium" if x < 0.03 else
        "High"
    )

    return df
