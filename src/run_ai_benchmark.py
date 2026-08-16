"""
Quantisense Capital
Phase 2 AI Benchmark
"""

from sklearn.metrics import accuracy_score

from ai_dataset import prepare_ai_dataset
from ai_features import prepare_features
from ai_train import walk_forward_splits
from ai_baseline import train_baseline
from ai_random_forest import train_random_forest
from ai_xgboost import train_xgboost
from ai_lightgbm import train_lightgbm
from ai_evaluation import compare_models


def evaluate_model(name, trainer, X, y):
    scores = []

    for X_train, X_test, y_train, y_test in walk_forward_splits(X, y):
        model = trainer(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(
            accuracy_score(y_test, predictions)
        )

    return {
        "mean_accuracy": sum(scores) / len(scores),
        "folds": scores
    }


def main():
    df = prepare_ai_dataset(
        "../data/BTC_USD_phase1.csv"
    )

    X, y = prepare_features(df)

    results = {
        "Logistic Regression": evaluate_model(
            "Logistic Regression",
            train_baseline,
            X,
            y
        ),
        "Random Forest": evaluate_model(
            "Random Forest",
            train_random_forest,
            X,
            y
        ),
        "XGBoost": evaluate_model(
            "XGBoost",
            train_xgboost,
            X,
            y
        ),
        "LightGBM": evaluate_model(
            "LightGBM",
            train_lightgbm,
            X,
            y
        )
    }

    print(compare_models(results).to_string())


if __name__ == "__main__":
    main()