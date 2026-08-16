"""
Quantisense Capital
Phase 2 AI Model Evaluation
"""

import pandas as pd


def compare_models(results):
    """
    Compare walk-forward model performance.
    """

    df = pd.DataFrame(results).T
    df = df.sort_values("mean_accuracy", ascending=False)

    return df