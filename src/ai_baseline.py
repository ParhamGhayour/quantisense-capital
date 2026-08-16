"""
Quantisense Capital
Phase 2 AI Baseline Model
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_baseline(X_train, y_train):
    """
    Train a scaled logistic regression baseline.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    return model