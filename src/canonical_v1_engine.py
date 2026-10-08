import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

DATA_DIR = "data"
N_SPLITS = 5

ASSETS = {
    "BTC": "BTC_USD_multi_year_phase1.csv",
    "ETH": "ETH_USD_multi_year_phase1.csv",
    "SOL": "SOL_USD_multi_year_phase1.csv",
    "SPY": "SPY_multi_year_phase1.csv",
    "QQQ": "QQQ_multi_year_phase1.csv",
}


def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Date
    date_col = next(
        (c for c in df.columns if c.lower() in ["date", "datetime", "timestamp"]),
        None
    )
    if date_col is None:
        raise ValueError(f"No date column in {path}")

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Required price
    close_col = next(
        (c for c in df.columns if c.lower() == "close"),
        None
    )
    if close_col is None:
        raise ValueError(f"No Close column in {path}")

    if close_col != "Close":
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    # Numeric OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_features(df):
    x = df.copy()

    # Causal price features
    x["Return_1D"] = x["Close"].pct_change(1)
    x["Return_3D"] = x["Close"].pct_change(3)
    x["Return_7D"] = x["Close"].pct_change(7)
    x["Return_14D"] = x["Close"].pct_change(14)
    x["Return_30D"] = x["Close"].pct_change(30)

    x["Momentum_Change"] = x["Return_1D"].diff()

    for n in [9, 21, 50, 100, 250]:
        ema = x["Close"].ewm(span=n, adjust=False).mean()
        x[f"EMA{n}"] = ema
        x[f"Close_EMA{n}_Ratio"] = x["Close"] / ema

    x["EMA9_EMA21_Ratio"] = x["EMA9"] / x["EMA21"]
    x["EMA21_EMA50_Ratio"] = x["EMA21"] / x["EMA50"]

    x["Volatility_7D"] = x["Return_1D"].rolling(7).std()
    x["Volatility_14D"] = x["Return_1D"].rolling(14).std()
    x["Volatility_30D"] = x["Return_1D"].rolling(30).std()

    if all(c in x.columns for c in ["High", "Low", "Close"]):
        prev_close = x["Close"].shift(1)

        tr = pd.concat(
            [
                x["High"] - x["Low"],
                (x["High"] - prev_close).abs(),
                (x["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        x["ATR_Percent"] = tr.rolling(14).mean() / x["Close"]

        x["OHLC_Range"] = (x["High"] - x["Low"]) / x["Close"]

        x["Upper_Wick"] = (
            x["High"] - x[["Open", "Close"]].max(axis=1)
        ) / x["Close"] if "Open" in x.columns else np.nan

        x["Lower_Wick"] = (
            x[["Open", "Close"]].min(axis=1) - x["Low"]
        ) / x["Close"] if "Open" in x.columns else np.nan

    if "Volume" in x.columns:
        x["Volume_Change"] = x["Volume"].pct_change()
        x["Relative_Volume"] = (
            x["Volume"] / x["Volume"].rolling(20).mean()
        )

    # Market score
    x["EMA_Score"] = (
        (x["Close"] > x["EMA9"]).astype(int)
        + (x["Close"] > x["EMA21"]).astype(int)
        + (x["Close"] > x["EMA50"]).astype(int)
        + (x["Close"] > x["EMA100"]).astype(int)
        + (x["Close"] > x["EMA250"]).astype(int)
    ) / 5.0

    # Target is strictly next-day
    x["Forward_Return_1D"] = x["Close"].shift(-1) / x["Close"] - 1
    x["Actual_Up"] = (x["Forward_Return_1D"] > 0).astype(float)

    feature_cols = [
        "Return_1D",
        "Return_3D",
        "Return_7D",
        "Return_14D",
        "Return_30D",
        "Momentum_Change",
        "Close_EMA9_Ratio",
        "Close_EMA21_Ratio",
        "Close_EMA50_Ratio",
        "Close_EMA100_Ratio",
        "Close_EMA250_Ratio",
        "EMA9_EMA21_Ratio",
        "EMA21_EMA50_Ratio",
        "Volatility_7D",
        "Volatility_14D",
        "Volatility_30D",
        "ATR_Percent",
        "OHLC_Range",
        "Upper_Wick",
        "Lower_Wick",
        "Volume_Change",
        "Relative_Volume",
        "EMA_Score",
    ]

    feature_cols = [c for c in feature_cols if c in x.columns]

    # Remove invalid rows only after all features are calculated
    x = x.dropna(subset=feature_cols + ["Forward_Return_1D"]).reset_index(drop=True)

    return x, feature_cols


def evaluate_asset(asset, filename):
    path = os.path.join(DATA_DIR, filename)

    df = load_data(path)
    df, features = build_features(df)

    X = df[features].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)

    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)

    y = df["Actual_Up"].astype(int)

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    rows = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight=None,
        )

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)

        test = df.iloc[test_idx].copy()

        # Position is based only on information available at day t.
        position = np.where(prob >= 0.55, 1.0,
                   np.where(prob <= 0.45, -1.0, 0.0))

        # Return from t -> t+1.
        gross_return = position * test["Forward_Return_1D"].values

        test["Probability"] = prob
        test["Prediction"] = pred
        test["Position"] = position
        test["Strategy_Return"] = gross_return
        test["Fold"] = fold

        for i in range(len(test)):
            rows.append(test.iloc[i])

        fold_return = np.prod(1 + gross_return) - 1

        fold_rows.append({
            "Asset": asset,
            "Fold": fold,
            "Observations": len(test),
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
            "Return": fold_return,
            "Mean_Daily_Return": np.mean(gross_return),
        })

    result = pd.DataFrame(rows)

    # Ensure chronological ordering
    result = result.sort_values("Date").reset_index(drop=True)

    return result, pd.DataFrame(fold_rows)


def calculate_metrics(result, cost_bps=0):
    r = result["Strategy_Return"].astype(float).values

    position = result["Position"].astype(float).values

    turnover = np.abs(np.diff(np.r_[0, position]))

    costs = turnover * cost_bps / 10000.0
    net = r - costs

    equity = np.cumprod(1 + net)

    cumulative = equity[-1] - 1

    if np.std(net, ddof=1) > 0:
        sharpe = (
            np.mean(net) / np.std(net, ddof=1)
        ) * np.sqrt(252)
    else:
        sharpe = np.nan

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    accuracy = accuracy_score(
        result["Actual_Up"].astype(int),
        result["Prediction"].astype(int),
    )

    balanced = balanced_accuracy_score(
        result["Actual_Up"].astype(int),
        result["Prediction"].astype(int),
    )

    return {
        "Observations": len(result),
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced,
        "Gross_Return": np.prod(1 + r) - 1,
        "Net_Return": cumulative,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Turnover": turnover.sum(),
        "Average_Daily_Turnover": turnover.mean(),
        "Cost_bps": cost_bps,
    }


def main():
    print("=" * 70)
    print("QUANTISENSE CAPITAL — CANONICAL V1 ENGINE")
    print("=" * 70)

    all_results = []
    all_folds = []

    for asset, filename in ASSETS.items():
        print(f"\n[{asset}] evaluating...")

        result, folds = evaluate_asset(asset, filename)

        result["Asset"] = asset

        all_results.append(result)
        all_folds.append(folds)

        print(
            f"  observations={len(result)} "
            f"accuracy={accuracy_score(result.Actual_Up, result.Prediction):.4f}"
        )

    all_results = pd.concat(all_results, ignore_index=True)
    all_folds = pd.concat(all_folds, ignore_index=True)

    # Save raw canonical predictions
    all_results.to_csv(
        os.path.join(DATA_DIR, "V1_CANONICAL_PREDICTIONS.csv"),
        index=False,
    )

    all_folds.to_csv(
        os.path.join(DATA_DIR, "V1_CANONICAL_FOLDS.csv"),
        index=False,
    )

    # Cost sensitivity
    metrics = []

    for asset in ASSETS:
        r = all_results[all_results["Asset"] == asset].copy()

        for cost in [0, 10, 25, 50]:
            m = calculate_metrics(r, cost)
            m["Asset"] = asset
            metrics.append(m)

    metrics = pd.DataFrame(metrics)

    metrics = metrics[
        [
            "Asset",
            "Cost_bps",
            "Observations",
            "Accuracy",
            "Balanced_Accuracy",
            "Gross_Return",
            "Net_Return",
            "Sharpe",
            "MaxDD",
            "Turnover",
            "Average_Daily_Turnover",
        ]
    ]

    metrics.to_csv(
        os.path.join(DATA_DIR, "V1_CANONICAL_RESULTS.csv"),
        index=False,
    )

    # Buy and hold comparison
    passive = []

    for asset in ASSETS:
        r = all_results[all_results["Asset"] == asset].copy()

        bh = r["Forward_Return_1D"].values

        equity = np.cumprod(1 + bh)
        ret = equity[-1] - 1

        sharpe = (
            np.mean(bh) / np.std(bh, ddof=1) * np.sqrt(252)
            if np.std(bh, ddof=1) > 0
            else np.nan
        )

        dd = equity / np.maximum.accumulate(equity) - 1

        passive.append({
            "Asset": asset,
            "Observations": len(r),
            "BuyHold_Return": ret,
            "BuyHold_Sharpe": sharpe,
            "BuyHold_MaxDD": dd.min(),
        })

    passive = pd.DataFrame(passive)

    passive.to_csv(
        os.path.join(DATA_DIR, "V1_CANONICAL_BUYHOLD.csv"),
        index=False,
    )

    print("\n" + "=" * 70)
    print("CANONICAL RESULTS — 0 bps")
    print("=" * 70)

    print(
        metrics[metrics["Cost_bps"] == 0][
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

    print("\n" + "=" * 70)
    print("CANONICAL RESULTS — 10 bps")
    print("=" * 70)

    print(
        metrics[metrics["Cost_bps"] == 10][
            [
                "Asset",
                "Net_Return",
                "Sharpe",
                "MaxDD",
                "Turnover",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("BUY & HOLD")
    print("=" * 70)

    print(passive.to_string(index=False))

    print("\n" + "=" * 70)
    print("CANONICAL V1 ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()