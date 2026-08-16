"""
Quantisense Capital
Phase 2 AI Regression Features
"""

from src.ai_features import prepare_features


def prepare_regression_features(df):
    """
    Prepare numerical features while preserving Next_Return
    as the regression target.
    """

    X, y_classification = prepare_features(
        df.rename(columns={"Next_Return": "Target"})
    )

    y = df["Next_Return"].copy()

    return X, y