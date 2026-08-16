import pandas as pd
import numpy as np


def add_market_score(df):
    """
    Quantisense composite market score
    Range: 0-100
    """

    df = df.copy()

    ema_score = (df["EMA_Score"] + 100) / 2

    volatility_score = np.where(
        df["Volatility_Regime"] == "Low",
        80,
        np.where(
            df["Volatility_Regime"] == "Medium",
            60,
            40
        )
    )

    atr_score = np.clip(
        100 - (df["ATR_Percent"] * 1000),
        0,
        100
    )

    df["Market_Score"] = (
        ema_score * 0.5
        + volatility_score * 0.25
        + atr_score * 0.25
    )

    df["Market_Score"] = df["Market_Score"].round(2)

    return df
