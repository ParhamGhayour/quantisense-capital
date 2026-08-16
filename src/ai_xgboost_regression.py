"""
Quantisense Capital
Phase 2 XGBoost Regression
"""

from xgboost import XGBRegressor


def train_xgboost_regression(X_train, y_train):
    """
    Train an XGBoost regressor for next-day return prediction.
    """

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model