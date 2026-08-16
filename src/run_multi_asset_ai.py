"""
Quantisense Capital
Phase 2 Multi-Asset AI Benchmark
"""

import os

from sklearn.metrics import accuracy_score

from ai_dataset import prepare_ai_dataset
from ai_features import prepare_features
from ai_train import walk_forward_splits
from ai_xgboost import train_xgboost


ASSETS = [
    "BTC_USD",
    "ETH_USD",
    "XRP_USD",
    "BNB_USD",
    "ADA_USD",
]


def evaluate_asset(asset):
    path = f"data/{asset}_phase1.csv"

    df = prepare_ai_dataset(path)
    X, y = prepare_features(df)

    scores = []

    for X_train, X_test, y_train, y_test in walk_forward_splits(X, y):
        model = train_xgboost(X_train, y_train)
        predictions = model.predict(X_test)

        scores.append(
            accuracy_score(y_test, predictions)
        )

    return {
        "mean_accuracy": sum(scores) / len(scores),
        "folds": scores
    }


def main():
    for asset in ASSETS:
        result = evaluate_asset(asset)

        print(
            asset,
            "mean=",
            round(result["mean_accuracy"], 4),
            "folds=",
            [round(x, 3) for x in result["folds"]]
        )


if __name__ == "__main__":
    main()