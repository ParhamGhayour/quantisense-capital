"""
Quantisense Capital
Phase 2 Multi-Asset Horizon Benchmark
"""

from src.ai_horizon_benchmark import evaluate_horizon_model


ASSETS = [
    "BTC_USD",
    "ETH_USD",
    "XRP_USD",
    "BNB_USD",
    "ADA_USD",
]


def main():
    results = {}

    for asset in ASSETS:
        path = f"data/{asset}_phase1.csv"

        result = evaluate_horizon_model(
            path,
            horizon=5,
        )

        results[asset] = result

        print(
            asset,
            "accuracy=",
            round(result["pooled_accuracy"], 4),
            "AUC=",
            round(result["auc"], 4),
            "folds=",
            [round(x, 3) for x in result["fold_accuracies"]],
        )

    mean_accuracy = sum(
        result["pooled_accuracy"]
        for result in results.values()
    ) / len(results)

    mean_auc = sum(
        result["auc"]
        for result in results.values()
    ) / len(results)

    print()
    print("Mean asset accuracy:", round(mean_accuracy, 4))
    print("Mean asset AUC:", round(mean_auc, 4))


if __name__ == "__main__":
    main()