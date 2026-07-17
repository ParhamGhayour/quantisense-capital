
"""
Quantisense Capital
Market Data Validation Engine
"""

import pandas as pd


def validate_market_data(data: pd.DataFrame) -> dict:
    """
    Validate OHLCV market data structure.
    """

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    result = {}

    result["rows"] = len(data)

    result["missing_values"] = data.isna().sum().sum()

    result["has_required_columns"] = all(
        col in data.columns.get_level_values(0)
        for col in required_columns
    )

    result["start_date"] = data.index.min()

    result["end_date"] = data.index.max()

    return result
