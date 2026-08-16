"""
Quantisense Capital
Phase 2 AI Dataset Preparation
"""

import pandas as pd


TARGET_COLUMN = "Target"


def prepare_ai_dataset(path):
    """
    Load Phase 1 data and create the next-day direction target.
    """

    df = pd.read_csv(path).copy()

    df[TARGET_COLUMN] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    df = df.iloc[:-1].copy()

    return df