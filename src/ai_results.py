"""
Quantisense Capital
Phase 2 AI Benchmark Results
"""

import pandas as pd


RESULTS = {
    "BTC_USD": 0.4931,
    "ETH_USD": 0.5103,
    "XRP_USD": 0.5138,
    "BNB_USD": 0.4793,
    "ADA_USD": 0.4897,
}


def benchmark_summary():
    df = pd.DataFrame.from_dict(
        RESULTS,
        orient="index",
        columns=["mean_accuracy"]
    )

    df.loc["ALL_ASSETS", "mean_accuracy"] = (
        df["mean_accuracy"].mean()
    )

    return df


if __name__ == "__main__":
    print(
        benchmark_summary().to_string()
    )