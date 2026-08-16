"""
Quantisense Capital
Phase 2 AI Feature Preparation
"""

import pandas as pd


TARGET_COLUMN = "Target"


def prepare_features(df):
    """
    Build scale-aware numerical ML features.

    Raw price levels are excluded because they are
    non-stationary and scale-dependent.
    """

    df = df.copy()

    # Scale-independent price/indicator relationships
    df["EMA9_to_Close"] = df["EMA_9"] / df["Close"]
    df["EMA21_to_Close"] = df["EMA_21"] / df["Close"]
    df["EMA50_to_Close"] = df["EMA_50"] / df["Close"]
    df["EMA100_to_Close"] = df["EMA_100"] / df["Close"]
    df["EMA250_to_Close"] = df["EMA_250"] / df["Close"]

    df["EMA9_to_EMA21"] = df["EMA_9"] / df["EMA_21"]
    df["EMA21_to_EMA50"] = df["EMA_21"] / df["EMA_50"]
    df["EMA50_to_EMA100"] = df["EMA_50"] / df["EMA_100"]
    df["EMA100_to_EMA250"] = df["EMA_100"] / df["EMA_250"]

    df["High_Low_Range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["Open_Close_Range"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    categorical_columns = [
        "Trend",
        "Volatility_Regime"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    excluded_columns = [
        TARGET_COLUMN,
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "EMA_9",
        "EMA_21",
        "EMA_50",
        "EMA_100",
        "EMA_250",
        "True_Range",
        "ATR"
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    y = df[TARGET_COLUMN]

    return X, y