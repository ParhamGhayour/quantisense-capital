"""
Quantisense Capital
Phase 2 Purged Horizon Benchmark
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ai_features import prepare_features
from src.ai_horizon_dataset import prepare_horizon_dataset
from src.ai_train import purged_walk_forward_splits
from src.ai_xgboost import train_xgboost


CORE_FEATURES = [
    "EMA_Score",
    "Market_Score",
    "Volatility",
    "ATR_Percent",
    "Return",
    "Log_Return",
]


def evaluate_horizon_model(path, horizon=5):
    df = prepare_horizon_dataset(path, horizon)

    # Prepare only features that are available at prediction time.
    feature_df = df.assign(Target=0)

    X_all, _ = prepare_features(
        feature_df.drop(columns=["Forward_Return"])
    )

    X_all = X_all.reset_index(drop=True)
    forward_returns = df["Forward_Return"].reset_index(drop=True)

    # Core Phase 1 signals only.
    X_all = X_all[CORE_FEATURES]

    pooled_true = []
    pooled_pred = []
    pooled_prob = []
    fold_scores = []
    thresholds = []

    for X_train, X_test, _, _ in purged_walk_forward_splits(
        X_all,
        forward_returns,
        horizon=horizon,
        n_splits=5,
    ):
        train_returns = forward_returns.iloc[X_train.index]
        test_returns = forward_returns.iloc[X_test.index]

        threshold = train_returns.median()

        y_train = (train_returns > threshold).astype(int)
        y_test = (test_returns > threshold).astype(int)

        model = train_xgboost(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fold_scores.append(
            accuracy_score(y_test, predictions)
        )

        thresholds.append(threshold)
        pooled_true.extend(y_test.tolist())
        pooled_pred.extend(predictions.tolist())
        pooled_prob.extend(probabilities.tolist())

    return {
        "fold_accuracies": fold_scores,
        "thresholds": thresholds,
        "pooled_accuracy": accuracy_score(
            pooled_true,
            pooled_pred,
        ),
        "auc": roc_auc_score(
            pooled_true,
            pooled_prob,
        ),
        "samples": len(pooled_true),
    }


if __name__ == "__main__":
    result = evaluate_horizon_model(
        "data/BTC_USD_phase1.csv",
        horizon=5,
    )

    print(
        "Fold accuracies:",
        [round(x, 3) for x in result["fold_accuracies"]],
    )

    print(
        "Thresholds:",
        [round(x, 5) for x in result["thresholds"]],
    )

    print(
        "Pooled accuracy:",
        round(result["pooled_accuracy"], 4),
    )

    print(
        "AUC:",
        round(result["auc"], 4),
    )

    print(
        "Samples:",
        result["samples"],
    )