"""
Quantisense Capital
Phase 2 Multi-Day AI Target
"""

import pandas as pd


def prepare_horizon_dataset(path, horizon=5):
    """
    Create a forward-return dataset.

    The classification threshold is deliberately NOT calculated here.
    It must be calculated from each training fold only.
    """

    df = pd.read_csv(path).copy()

    df["Forward_Return"] = (
        df["Close"].shift(-horizon) / df["Close"] - 1
    )

    df = df.iloc[:-horizon].copy()

    return df