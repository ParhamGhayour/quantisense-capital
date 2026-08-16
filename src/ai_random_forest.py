"""
Quantisense Capital
Phase 2 Random Forest Baseline
"""

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model