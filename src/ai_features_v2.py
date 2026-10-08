"""
Quantisense Capital
Phase 2.1 AI Feature Preparation
"""

import pandas as pd


TARGET_COLUMNS = [
    "Target_1D",
    "Target_3D",
    "Target_7D_Return",
]


def prepare_features_v2(df):
    """
    Build causal, scale-independent features for ML 2.0.

    All features use information available on the current
    observation or earlier observations only.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Returns / momentum
    # ---------------------------------------------------------

    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_3D"] = df["Close"].pct_change(3)
    df["Return_7D"] = df["Close"].pct_change(7)
    df["Return_14D"] = df["Close"].pct_change(14)
    df["Return_30D"] = df["Close"].pct_change(30)

    # Momentum acceleration
    df["Momentum_3D_Change"] = df["Return_3D"].diff()
    df["Momentum_7D_Change"] = df["Return_7D"].diff()

    # ---------------------------------------------------------
    # EMA structure
    # ---------------------------------------------------------

    ema_periods = [9, 21, 50, 100, 250]

    for period in ema_periods:
        df[f"Close_to_EMA{period}"] = (
            df["Close"] / df[f"EMA_{period}"] - 1.0
        )

    df["EMA9_to_EMA21"] = (
        df["EMA_9"] / df["EMA_21"] - 1.0
    )

    df["EMA21_to_EMA50"] = (
        df["EMA_21"] / df["EMA_50"] - 1.0
    )

    df["EMA50_to_EMA100"] = (
        df["EMA_50"] / df["EMA_100"] - 1.0
    )

    df["EMA100_to_EMA250"] = (
        df["EMA_100"] / df["EMA_250"] - 1.0
    )

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    df["Rolling_Volatility_7D"] = (
        df["Return_1D"].rolling(7).std()
    )

    df["Rolling_Volatility_14D"] = (
        df["Return_1D"].rolling(14).std()
    )

    df["Rolling_Volatility_30D"] = (
        df["Return_1D"].rolling(30).std()
    )

    df["ATR_Percent_Current"] = df["ATR_Percent"]

    df["Volatility_Expansion"] = (
        df["Rolling_Volatility_7D"]
        / df["Rolling_Volatility_30D"]
    )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    df["Volume_Change_1D"] = df["Volume"].pct_change()

    volume_mean_20 = df["Volume"].rolling(20).mean()

    df["Relative_Volume_20D"] = (
        df["Volume"] / volume_mean_20
    )

    # ---------------------------------------------------------
    # Candle structure
    # ---------------------------------------------------------

    df["High_Low_Range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["Open_Close_Range"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    df["Upper_Wick"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / df["Close"]

    df["Lower_Wick"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    ) / df["Close"]

    # ---------------------------------------------------------
    # Phase 1 information
    # ---------------------------------------------------------

    df["Market_Score_Change_1D"] = (
        df["Market_Score"].diff()
    )

    df["Market_Score_Change_7D"] = (
        df["Market_Score"].diff(7)
    )

    df["EMA_Score_Change_1D"] = (
        df["EMA_Score"].diff()
    )

    df["EMA_Score_Change_7D"] = (
        df["EMA_Score"].diff(7)
    )

    # ---------------------------------------------------------
    # Encode categorical Phase 1 states
    # ---------------------------------------------------------

    categorical_columns = [
        "Trend",
        "Volatility_Regime",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    # ---------------------------------------------------------
    # Remove targets and raw/non-feature columns
    # ---------------------------------------------------------

    excluded_columns = [
        *TARGET_COLUMNS,
        "Date",
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
        "ATR",
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    return X