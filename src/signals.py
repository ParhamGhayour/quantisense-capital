
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
    Safely retrieve a column from either normal
    or MultiIndex DataFrames.
    """

    if isinstance(df.columns, pd.MultiIndex):
        column = df[name]

        if isinstance(column, pd.DataFrame):
            return column.iloc[:, 0]

        return column

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
    Classify ATR volatility using expanding historical quantiles.

    Each row is classified using only information available
    up to that row.
    """

    df = df.copy()

    atr_percent = df["ATR_Percent"]

    expanding = atr_percent.expanding(min_periods=30)

    low_threshold = expanding.quantile(0.33)
    high_threshold = expanding.quantile(0.67)

    df["Volatility_Regime"] = "Medium"

    df.loc[
        atr_percent <= low_threshold,
        "Volatility_Regime"
    ] = "Low"

    df.loc[
        atr_percent >= high_threshold,
        "Volatility_Regime"
    ] = "High"

    return df