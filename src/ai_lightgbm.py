"""
Quantisense Capital
Phase 2 LightGBM Model
"""

from lightgbm import LGBMClassifier


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    return model