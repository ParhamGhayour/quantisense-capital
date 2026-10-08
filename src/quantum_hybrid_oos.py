"""
Quantisense Capital
Trained Quantum Hybrid OOS Experiment - V1

A small variational quantum classifier evaluated
chronologically.

IMPORTANT:
This is an experimental research module.
It does not assume or claim quantum advantage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_hybrid_intelligence.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_quantum_hybrid_oos.csv"
)


FEATURES = [
    "Market_Probability",
    "Price_EMI",
    "Media_Score",
    "Market_Media_Agreement",
]


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_data(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df["Target"] = pd.to_numeric(
        df["Actual_Up"],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    ).copy()

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    return df


# ==========================================================
# FEATURE ENCODING
# ==========================================================

def normalize_features(X):

    X = np.asarray(
        X,
        dtype=float
    )

    return np.clip(
        X / 100.0,
        0.0,
        1.0
    )


# ==========================================================
# QUANTUM CIRCUIT
# ==========================================================

def build_circuit(
    features,
    weights
):

    features = np.asarray(
        features,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    angles = (
        features * np.pi
    )

    qc = QuantumCircuit(
        4,
        1
    )

    # Feature encoding

    for qubit in range(4):

        qc.ry(
            float(angles[qubit]),
            qubit
        )

    # Variational layer

    for qubit in range(4):

        qc.ry(
            float(weights[qubit]),
            qubit
        )

    # Entanglement

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    # Measure first qubit

    qc.measure(
        0,
        0
    )

    return qc


# ==========================================================
# QUANTUM PROBABILITY
# ==========================================================

def quantum_probability(
    features,
    weights,
    shots=256
):

    qc = build_circuit(
        features,
        weights
    )

    simulator = AerSimulator()

    result = simulator.run(
        qc,
        shots=shots
    ).result()

    counts = result.get_counts()

    ones = counts.get(
        "1",
        0
    )

    total = sum(
        counts.values()
    )

    if total == 0:

        return 0.5

    return (
        ones / total
    )


# ==========================================================
# TRAIN QUANTUM PARAMETERS
# ==========================================================

def train_quantum_model(
    X_train,
    y_train,
    epochs=8,
    learning_rate=0.15,
):

    rng = np.random.default_rng(
        42
    )

    weights = rng.normal(
        0,
        0.2,
        size=4
    )

    # Finite-difference parameter optimization.
    #
    # This is intentionally simple for V1.
    # It keeps the experiment transparent and
    # avoids introducing a large optimizer stack.

    for epoch in range(epochs):

        predictions = np.array(
            [
                quantum_probability(
                    x,
                    weights,
                    shots=128
                )
                for x in X_train
            ]
        )

        predictions = np.clip(
            predictions,
            1e-6,
            1 - 1e-6
        )

        base_loss = -np.mean(
            y_train
            * np.log(predictions)
            + (
                1 - y_train
            )
            * np.log(1 - predictions)
        )

        gradients = np.zeros(
            4
        )

        epsilon = 0.05

        for parameter in range(4):

            weights_plus = (
                weights.copy()
            )

            weights_minus = (
                weights.copy()
            )

            weights_plus[
                parameter
            ] += epsilon

            weights_minus[
                parameter
            ] -= epsilon

            plus_predictions = np.array(
                [
                    quantum_probability(
                        x,
                        weights_plus,
                        shots=128
                    )
                    for x in X_train
                ]
            )

            minus_predictions = np.array(
                [
                    quantum_probability(
                        x,
                        weights_minus,
                        shots=128
                    )
                    for x in X_train
                ]
            )

            plus_predictions = np.clip(
                plus_predictions,
                1e-6,
                1 - 1e-6
            )

            minus_predictions = np.clip(
                minus_predictions,
                1e-6,
                1 - 1e-6
            )

            plus_loss = -np.mean(
                y_train
                * np.log(
                    plus_predictions
                )
                + (
                    1 - y_train
                )
                * np.log(
                    1 - plus_predictions
                )
            )

            minus_loss = -np.mean(
                y_train
                * np.log(
                    minus_predictions
                )
                + (
                    1 - y_train
                )
                * np.log(
                    1 - minus_predictions
                )
            )

            gradients[
                parameter
            ] = (
                plus_loss
                - minus_loss
            ) / (
                2 * epsilon
            )

        weights -= (
            learning_rate
            * gradients
        )

        print(
            f"      Epoch {epoch + 1}/{epochs} "
            f"| Loss={base_loss:.4f}"
        )

    return weights


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("QUANTISENSE TRAINED QUANTUM OOS")
    print("=" * 70)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    raw = pd.read_csv(
        INPUT_PATH
    )

    print()
    print(
        f"Raw rows: {len(raw)}"
    )

    df = prepare_data(
        raw
    )

    print(
        f"Usable rows: {len(df)}"
    )

    # ------------------------------------------------------
    # Time-series folds
    # ------------------------------------------------------

    n_splits = 3

    fold_size = (
        len(df)
        // (n_splits + 1)
    )

    results = []

    all_classical = []
    all_quantum = []
    all_hybrid = []
    all_actual = []

    # ------------------------------------------------------
    # Walk forward
    # ------------------------------------------------------

    for fold in range(
        n_splits
    ):

        train_end = (
            fold_size
            * (fold + 1)
        )

        test_start = train_end

        test_end = (
            fold_size
            * (fold + 2)
        )

        train = df.iloc[
            :train_end
        ].copy()

        test = df.iloc[
            test_start:test_end
        ].copy()

        if len(test) == 0:

            continue

        print()
        print("=" * 70)
        print(
            f"FOLD {fold + 1}"
        )
        print("=" * 70)

        print(
            f"Train rows: {len(train)}"
        )

        print(
            f"Test rows:  {len(test)}"
        )

        # --------------------------------------------------
        # Features
        # --------------------------------------------------

        X_train = normalize_features(
            train[FEATURES].values
        )

        X_test = normalize_features(
            test[FEATURES].values
        )

        y_train = (
            train["Target"]
            .values
        )

        y_test = (
            test["Target"]
            .values
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        print()
        print(
            "Training quantum parameters..."
        )

        weights = train_quantum_model(
            X_train,
            y_train,
            epochs=8,
            learning_rate=0.15,
        )

        # --------------------------------------------------
        # OOS prediction
        # --------------------------------------------------

        print()
        print(
            "Generating OOS quantum predictions..."
        )

        quantum_probabilities = np.array(
            [
                quantum_probability(
                    x,
                    weights,
                    shots=256
                )
                for x in X_test
            ]
        )

        # --------------------------------------------------
        # Classical probability
        # --------------------------------------------------

        classical_probabilities = (
            test["Market_Probability"]
            .values
            / 100.0
        )

        # --------------------------------------------------
        # Hybrid
        # --------------------------------------------------

        hybrid_probabilities = (
            0.70
            * classical_probabilities

            + 0.30
            * quantum_probabilities
        )

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        classical_predictions = (
            classical_probabilities
            >= 0.50
        ).astype(int)

        quantum_predictions = (
            quantum_probabilities
            >= 0.50
        ).astype(int)

        hybrid_predictions = (
            hybrid_probabilities
            >= 0.50
        ).astype(int)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        classical_accuracy = (
            accuracy_score(
                y_test,
                classical_predictions
            )
        )

        quantum_accuracy = (
            accuracy_score(
                y_test,
                quantum_predictions
            )
        )

        hybrid_accuracy = (
            accuracy_score(
                y_test,
                hybrid_predictions
            )
        )

        classical_balanced = (
            balanced_accuracy_score(
                y_test,
                classical_predictions
            )
        )

        quantum_balanced = (
            balanced_accuracy_score(
                y_test,
                quantum_predictions
            )
        )

        hybrid_balanced = (
            balanced_accuracy_score(
                y_test,
                hybrid_predictions
            )
        )

        print()
        print(
            f"Classical accuracy: "
            f"{classical_accuracy:.4f}"
        )

        print(
            f"Quantum accuracy:   "
            f"{quantum_accuracy:.4f}"
        )

        print(
            f"Hybrid accuracy:    "
            f"{hybrid_accuracy:.4f}"
        )

        print()

        print(
            f"Classical balanced: "
            f"{classical_balanced:.4f}"
        )

        print(
            f"Quantum balanced:   "
            f"{quantum_balanced:.4f}"
        )

        print(
            f"Hybrid balanced:    "
            f"{hybrid_balanced:.4f}"
        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        fold_results = test[
            [
                "Date",
                "Target",
            ]
        ].copy()

        fold_results[
            "Fold"
        ] = fold + 1

        fold_results[
            "Classical_Probability"
        ] = classical_probabilities

        fold_results[
            "Quantum_Probability"
        ] = quantum_probabilities

        fold_results[
            "Hybrid_Probability"
        ] = hybrid_probabilities

        fold_results[
            "Classical_Prediction"
        ] = classical_predictions

        fold_results[
            "Quantum_Prediction"
        ] = quantum_predictions

        fold_results[
            "Hybrid_Prediction"
        ] = hybrid_predictions

        results.append(
            fold_results
        )

        all_classical.extend(
            classical_predictions
        )

        all_quantum.extend(
            quantum_predictions
        )

        all_hybrid.extend(
            hybrid_predictions
        )

        all_actual.extend(
            y_test
        )

    # ======================================================
    # FINAL OOS RESULTS
    # ======================================================

    output = pd.concat(
        results,
        ignore_index=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    actual = np.asarray(
        all_actual
    )

    classical = np.asarray(
        all_classical
    )

    quantum = np.asarray(
        all_quantum
    )

    hybrid = np.asarray(
        all_hybrid
    )

    # ------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------

    classical_accuracy = (
        accuracy_score(
            actual,
            classical
        )
    )

    quantum_accuracy = (
        accuracy_score(
            actual,
            quantum
        )
    )

    hybrid_accuracy = (
        accuracy_score(
            actual,
            hybrid
        )
    )

    classical_balanced = (
        balanced_accuracy_score(
            actual,
            classical
        )
    )

    quantum_balanced = (
        balanced_accuracy_score(
            actual,
            quantum
        )
    )

    hybrid_balanced = (
        balanced_accuracy_score(
            actual,
            hybrid
        )
    )

    # ======================================================
    # REPORT
    # ======================================================

    print()
    print("=" * 70)
    print("FINAL QUANTUM OOS RESULTS")
    print("=" * 70)

    print()
    print(
        f"OOS observations: "
        f"{len(output)}"
    )

    print()
    print("ACCURACY")
    print("-" * 70)

    print(
        f"Classical: "
        f"{classical_accuracy:.4f}"
    )

    print(
        f"Quantum:   "
        f"{quantum_accuracy:.4f}"
    )

    print(
        f"Hybrid:    "
        f"{hybrid_accuracy:.4f}"
    )

    print()
    print("BALANCED ACCURACY")
    print("-" * 70)

    print(
        f"Classical: "
        f"{classical_balanced:.4f}"
    )

    print(
        f"Quantum:   "
        f"{quantum_balanced:.4f}"
    )

    print(
        f"Hybrid:    "
        f"{hybrid_balanced:.4f}"
    )

    print()
    print("DELTA VS CLASSICAL")
    print("-" * 70)

    print(
        f"Quantum accuracy delta: "
        f"{quantum_accuracy - classical_accuracy:+.4f}"
    )

    print(
        f"Hybrid accuracy delta:  "
        f"{hybrid_accuracy - classical_accuracy:+.4f}"
    )

    print(
        f"Quantum balanced delta: "
        f"{quantum_balanced - classical_balanced:+.4f}"
    )

    print(
        f"Hybrid balanced delta:  "
        f"{hybrid_balanced - classical_balanced:+.4f}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()