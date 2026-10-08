import pandas as pd
import numpy as np
import os

DATA = "data"

results = pd.read_csv(f"{DATA}/V1_CANONICAL_RESULTS.csv")
folds = pd.read_csv(f"{DATA}/V1_CANONICAL_FOLDS.csv")
bh = pd.read_csv(f"{DATA}/V1_CANONICAL_BUYHOLD.csv")

print("=" * 75)
print("QUANTISENSE CAPITAL — CANONICAL V1 FINAL AUDIT")
print("=" * 75)

print("\nCANONICAL PERFORMANCE")
print("-" * 75)

base = results[results["Cost_bps"] == 0].copy()

print(
    base[
        [
            "Asset",
            "Observations",
            "Accuracy",
            "Balanced_Accuracy",
            "Net_Return",
            "Sharpe",
            "MaxDD",
        ]
    ].to_string(index=False)
)

print("\n10 BPS EXECUTION")
print("-" * 75)

cost10 = results[results["Cost_bps"] == 10].copy()

print(
    cost10[
        [
            "Asset",
            "Net_Return",
            "Sharpe",
            "MaxDD",
            "Turnover",
        ]
    ].to_string(index=False)
)

print("\nFOLD CONSISTENCY")
print("-" * 75)

fold_summary = (
    folds.groupby("Asset")
    .agg(
        Folds=("Fold", "count"),
        Positive_Folds=("Return", lambda x: int((x > 0).sum())),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_Return=("Return", "mean"),
        Std_Return=("Return", "std"),
    )
    .reset_index()
)

print(fold_summary.to_string(index=False))

print("\nBUY & HOLD")
print("-" * 75)

print(
    bh[
        [
            "Asset",
            "BuyHold_Return",
            "BuyHold_Sharpe",
            "BuyHold_MaxDD",
        ]
    ].to_string(index=False)
)

print("\nCOST SURVIVAL")
print("-" * 75)

for asset in results["Asset"].unique():
    x = results[results["Asset"] == asset]

    survivors = x[x["Net_Return"] > 0]["Cost_bps"].tolist()

    if survivors:
        print(
            f"{asset}: profitable through "
            f"{max(survivors)} bps"
        )
    else:
        print(f"{asset}: not profitable at tested costs")

print("\nCANONICAL SCIENTIFIC VERDICT")
print("-" * 75)

positive_10 = int(
    (cost10["Net_Return"] > 0).sum()
)

print(
    f"Assets profitable at 10 bps: "
    f"{positive_10}/{len(cost10)}"
)

print(
    "\nThe canonical V1 system demonstrates only a modest "
    "directional edge and does not establish market-beating "
    "performance."
)

print(
    "\nBTC is the strongest candidate for further research, "
    "but its advantage is highly sensitive to transaction costs."
)

print(
    "\nBuy-and-hold produces substantially higher cumulative "
    "returns across the tested historical samples."
)

print(
    "\nThe appropriate V1 interpretation is selective/risk-aware "
    "market intelligence, not proven alpha generation."
)

print(
    "\nHistorical media coverage remains insufficient to validate "
    "the media intelligence layer."
)

print(
    "\nQuantum experiments do not establish quantum advantage."
)

print("\n" + "=" * 75)
print("V1 CANONICAL BASELINE ESTABLISHED")
print("=" * 75)

# Save final summary
summary = base.merge(
    bh,
    on=["Asset", "Observations"],
    how="left"
)

summary.to_csv(
    f"{DATA}/V1_CANONICAL_FINAL_SUMMARY.csv",
    index=False
)

with open(
    f"{DATA}/V1_CANONICAL_FINAL_REPORT.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("QUANTISENSE CAPITAL — V1 CANONICAL FINAL REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        "V1 status: OPERATIONAL RESEARCH PROTOTYPE\n"
    )

    f.write(
        "Market-beating claim: NOT ESTABLISHED\n"
    )

    f.write(
        "Quantum advantage: NOT ESTABLISHED\n"
    )

    f.write(
        "Media intelligence validation: INSUFFICIENT DATA\n\n"
    )

    f.write(
        base.to_string(index=False)
    )

    f.write("\n\n10 BPS RESULTS\n")
    f.write(cost10.to_string(index=False))

    f.write("\n\nBUY AND HOLD\n")
    f.write(bh.to_string(index=False))

print(
    f"\nSaved: {DATA}/V1_CANONICAL_FINAL_SUMMARY.csv"
)

print(
    f"Saved: {DATA}/V1_CANONICAL_FINAL_REPORT.txt"
)