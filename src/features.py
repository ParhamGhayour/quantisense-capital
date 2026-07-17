
"""
Quantisense Capital
Feature Engineering Engine
"""

import pandas as pd


def add_daily_return(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily percentage return feature.
    """

    data = data.copy()

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    return data
