"""
Quantisense Capital
Phase 2 AI Regression Baseline
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def train_regression_baseline(X_train, y_train):
    """
    Train a scaled Ridge regression baseline.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    return model