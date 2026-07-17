
"""
Quantisense Capital
Data Storage Engine
"""

import pandas as pd
import os


def save_dataset(
    data: pd.DataFrame,
    path: str
):
    """
    Save dataframe to CSV.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    data.to_csv(path)

    return path
