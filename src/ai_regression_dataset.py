"""
Quantisense Capital
Phase 2 AI Regression Dataset
"""

import pandas as pd


TARGET_COLUMN = "Next_Return"


def prepare_regression_dataset(path):
    """
    Create a next-day percentage-return target.
    """

    df = pd.read_csv(path).copy()

    df[TARGET_COLUMN] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    df = df.iloc[:-1].copy()

    return df