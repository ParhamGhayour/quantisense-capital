"""
Quantisense Capital
Canonical ML Engine — Phase 1/2.1

Pipeline:
    Market data
        ↓
    Phase 1 features
        ↓
    AI v2 features
        ↓
    Next-day target
        ↓
    Walk-forward Random Forest
        ↓
    OOS probability
        ↓
    BUY / HOLD / SELL signal
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

from src.ai_features_v2 import prepare_features_v2
from src.indicators import (
    add_returns,
    add_volatility,
    add_ema_indicators,
)
from src.atr import add_true_range, add_atr
from src.signals import add_ema_score, add_volatility_regime
from src.scoring import add_market_score


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_SPLITS = 5

RF_ESTIMATORS = 300
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_LEAF = 5
RF_RANDOM_STATE = 42


# ---------------------------------------------------------
# Phase 1 feature construction
# ---------------------------------------------------------

def build_phase1_features(df):
    """
    Build the existing causal Phase 1 market features.
    """

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna(how="all").copy()

    data = add_ema_indicators(data)
    data = add_returns(data)
    data = add_volatility(data)
    data = add_true_range(data)
    data = add_atr(data)
    data = add_ema_score(data)
    data = add_volatility_regime(data)
    data = add_market_score(data)

    return data


# ---------------------------------------------------------
# Target construction
# ---------------------------------------------------------

def add_target(data):
    """
    Create the next-day direction target.

    1 = next-day close is higher
    0 = next-day close is not higher
    """

    data = data.copy()

    data["Target_1D"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype("Int64")

    return data


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

def prepare_ml_data(data):
    """
    Build AI v2 features and align them with Target_1D.
    """

    data = add_target(data)

    features = prepare_features_v2(data)

    valid_rows = (
        features
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
        & data["Target_1D"].notna()
    )

    features = features.loc[valid_rows].copy()

    targets = data.loc[
        valid_rows,
        "Target_1D"
    ].astype(int)

    dates = data.loc[
        valid_rows,
        "Date"
    ].copy()

    close = data.loc[
        valid_rows,
        "Close"
    ].copy()

    future_close = (
        data["Close"]
        .shift(-1)
        .loc[valid_rows]
        .copy()
    )

    return (
        features,
        targets,
        dates,
        close,
        future_close,
    )


# ---------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------

def walk_forward_splits(X, y, n_splits=5):
    """
    Chronological walk-forward validation.
    """

    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    for train_idx, test_idx in splitter.split(X):

        yield (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            test_idx,
        )


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train the canonical Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------
# Signal
# ---------------------------------------------------------

def probability_to_signal(probability):
    """
    Convert model probability into a neutral baseline signal.

    This is deliberately conservative:
        probability >= 0.55 → BUY
        probability <= 0.45 → SELL
        otherwise → HOLD
    """

    if probability >= 0.55:
        return "BUY"

    if probability <= 0.45:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------
# Canonical OOS evaluation
# ---------------------------------------------------------

def run_oos_engine(data, n_splits=N_SPLITS):
    """
    Run the complete walk-forward ML engine.
    """

    (
        X,
        y,
        dates,
        close,
        future_close,
    ) = prepare_ml_data(data)

    all_results = []

    for fold, (
        X_train,
        X_test,
        y_train,
        y_test,
        test_idx,
    ) in enumerate(
        walk_forward_splits(
            X,
            y,
            n_splits=n_splits,
        ),
        start=1,
    ):

        model = train_model(
            X_train,
            y_train,
        )

        probability = model.predict_proba(
            X_test
        )[:, 1]

        prediction = (
            probability >= 0.50
        ).astype(int)

        fold_results = pd.DataFrame({
            "Date": dates.iloc[test_idx].values,
            "Close": close.iloc[test_idx].values,
            "Future_Close_1D": (
                future_close.iloc[test_idx].values
            ),
            "Actual_Up": (
                y_test.values
            ),
            "RF_Probability_Up": probability,
            "RF_Prediction": prediction,
        })

        fold_results["Forward_Return_1D"] = (
            fold_results["Future_Close_1D"]
            / fold_results["Close"]
            - 1.0
        )

        fold_results["RF_Confidence"] = (
            fold_results["RF_Probability_Up"]
            - 0.50
        ).abs()

        fold_results["Signal"] = (
            fold_results["RF_Probability_Up"]
            .apply(probability_to_signal)
        )

        fold_results["Correct"] = (
            fold_results["RF_Prediction"]
            == fold_results["Actual_Up"]
        )

        fold_results["Fold"] = fold

        all_results.append(
            fold_results
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    return results


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def summarize_results(results):
    """
    Produce a compact OOS performance summary.
    """

    accuracy = results["Correct"].mean()

    return {
        "observations": len(results),
        "accuracy": accuracy,
        "mean_forward_return": (
            results["Forward_Return_1D"].mean()
        ),
        "buy_signals": int(
            (results["Signal"] == "BUY").sum()
        ),
        "hold_signals": int(
            (results["Signal"] == "HOLD").sum()
        ),
        "sell_signals": int(
            (results["Signal"] == "SELL").sum()
        ),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("QUANTISENSE CAPITAL")
    print("CANONICAL ML ENGINE")
    print("=" * 60)

    input_path = (
        BASE_DIR
        / "data"
        / "BTC_USD_multi_year_phase1.csv"
    )

    output_path = (
        BASE_DIR
        / "data"
        / "BTC_USD_quantisense_oos.csv"
    )

    print(f"Loading: {input_path}")

    raw = pd.read_csv(
        input_path
    )

    raw["Date"] = pd.to_datetime(
        raw["Date"]
    )

    raw = raw.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    print(
        f"Input observations: {len(raw)}"
    )

    print("\nBuilding Phase 1 features...")

    data = build_phase1_features(
        raw
    )

    print(
        f"Feature observations: {len(data)}"
    )

    print("\nRunning walk-forward OOS engine...")

    results = run_oos_engine(
        data
    )

    summary = summarize_results(
        results
    )

    print("\n" + "=" * 60)
    print("OOS RESULT")
    print("=" * 60)

    print(
        f"Observations:       "
        f"{summary['observations']}"
    )

    print(
        f"Accuracy:           "
        f"{summary['accuracy']:.4f}"
    )

    print(
        f"Mean forward return:"
        f" {summary['mean_forward_return']:.4%}"
    )

    print(
        f"BUY signals:        "
        f"{summary['buy_signals']}"
    )

    print(
        f"HOLD signals:       "
        f"{summary['hold_signals']}"
    )

    print(
        f"SELL signals:       "
        f"{summary['sell_signals']}"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)