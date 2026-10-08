"""
Quantisense Capital
Quantum Hybrid Intelligence - V1

Small quantum classifier using Qiskit.

The experiment uses four compact, normalized features:

    1. Market probability
    2. Price EMI
    3. Media score
    4. Market/media agreement

The quantum circuit is intentionally small so that the
experiment remains reproducible on a local simulator.

IMPORTANT:
This module is an experimental research component.
It does NOT claim quantum advantage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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
    / "BTC_USD_quantum_hybrid.csv"
)


FEATURES = [
    "Market_Probability",
    "Price_EMI",
    "Media_Score",
    "Market_Media_Agreement",
]


# ==========================================================
# QUANTUM FEATURE ENCODING
# ==========================================================

def encode_features(features):

    """
    Encode four normalized features into four qubits.

    Each feature is mapped to an Ry rotation.

    Values are expected to be 0-100.
    """

    normalized = np.clip(
        np.asarray(features, dtype=float) / 100.0,
        0.0,
        1.0,
    )

    angles = normalized * np.pi

    circuit = QuantumCircuit(4, 4)

    for qubit, angle in enumerate(angles):

        circuit.ry(
            float(angle),
            qubit
        )

    return circuit


# ==========================================================
# ENTANGLEMENT
# ==========================================================

def add_entanglement(circuit):

    """
    Create a simple ring of entanglement.
    """

    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 0)

    return circuit


# ==========================================================
# QUANTUM CLASSIFIER
# ==========================================================

def quantum_probability(features):

    """
    Generate a quantum probability estimate.

    The probability is based on the probability of
    measuring the first qubit in |1>.
    """

    circuit = encode_features(
        features
    )

    circuit = add_entanglement(
        circuit
    )

    circuit.measure(
        range(4),
        range(4)
    )

    simulator = AerSimulator()

    result = simulator.run(
        circuit,
        shots=256,
    ).result()

    counts = result.get_counts()

    total = sum(
        counts.values()
    )

    ones = 0

    for bitstring, count in counts.items():

        # Qiskit returns classical bits in reverse order.
        # The rightmost bit corresponds to qubit 0.

        if bitstring[-1] == "1":

            ones += count

    probability = (
        ones / total
        if total > 0
        else 0.5
    )

    return float(probability)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("QUANTISENSE QUANTUM HYBRID INTELLIGENCE")
    print("=" * 70)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print()
    print(
        f"Rows loaded: {len(df)}"
    )

    # ------------------------------------------------------
    # Validate features
    # ------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing quantum features: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[FEATURES] = (
        df[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(50.0)
    )

    # ------------------------------------------------------
    # Quantum probability
    # ------------------------------------------------------

    print()
    print(
        "Running quantum circuit..."
    )

    quantum_probabilities = []

    for index, row in df.iterrows():

        probability = quantum_probability(
            [
                row[feature]
                for feature in FEATURES
            ]
        )

        quantum_probabilities.append(
            probability
        )

        if (
            (index + 1) % 250 == 0
            or index == len(df) - 1
        ):

            print(
                f"Quantum rows processed: "
                f"{index + 1}/{len(df)}"
            )

    df["Quantum_Probability_Up"] = (
        quantum_probabilities
    )

    df["Quantum_Score"] = (
        df["Quantum_Probability_Up"]
        * 100.0
    )

    # ------------------------------------------------------
    # Classical probability
    # ------------------------------------------------------

    if "Market_Probability" in df.columns:

        classical_probability = (
            df["Market_Probability"]
            / 100.0
        )

    else:

        classical_probability = (
            pd.Series(
                0.5,
                index=df.index
            )
        )

    # ------------------------------------------------------
    # Quantum / classical hybrid
    #
    # Classical model = 70%
    # Quantum model   = 30%
    # ------------------------------------------------------

    df["Quantum_Hybrid_Probability"] = (
        0.70
        * classical_probability

        + 0.30
        * df["Quantum_Probability_Up"]
    )

    df["Quantum_Hybrid_Score"] = (
        df["Quantum_Hybrid_Probability"]
        * 100.0
    )

    # ------------------------------------------------------
    # Signals
    # ------------------------------------------------------

    df["Quantum_Signal"] = np.select(
        [
            df["Quantum_Score"] >= 70,
            df["Quantum_Score"] >= 55,
            df["Quantum_Score"] <= 30,
            df["Quantum_Score"] <= 45,
        ],
        [
            "STRONG BUY",
            "BUY",
            "STRONG SELL",
            "SELL",
        ],
        default="HOLD",
    )

    df["Quantum_Hybrid_Signal"] = np.select(
        [
            df["Quantum_Hybrid_Score"] >= 70,
            df["Quantum_Hybrid_Score"] >= 55,
            df["Quantum_Hybrid_Score"] <= 30,
            df["Quantum_Hybrid_Score"] <= 45,
        ],
        [
            "STRONG BUY",
            "BUY",
            "STRONG SELL",
            "SELL",
        ],
        default="HOLD",
    )

    # ------------------------------------------------------
    # Agreement
    # ------------------------------------------------------

    df["Classical_Quantum_Agreement"] = (
        100.0
        - (
            df["Market_Probability"]
            - df["Quantum_Score"]
        ).abs()
    ).clip(0, 100)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("QUANTUM HYBRID RESULTS")
    print("=" * 70)

    print()
    print(
        f"Rows processed:              {len(df)}"
    )

    print(
        f"Mean quantum probability:     "
        f"{df['Quantum_Probability_Up'].mean():.4f}"
    )

    print(
        f"Mean classical probability:   "
        f"{classical_probability.mean():.4f}"
    )

    print(
        f"Mean quantum-hybrid score:     "
        f"{df['Quantum_Hybrid_Score'].mean():.2f}"
    )

    print(
        f"Mean classical/quantum "
        f"agreement:                  "
        f"{df['Classical_Quantum_Agreement'].mean():.2f}"
    )

    if len(df) > 0:

        latest = df.iloc[-1]

        print()
        print("LATEST QUANTUM STATE")
        print("-" * 70)

        print(
            f"Date:                       "
            f"{latest.get('Date', 'N/A')}"
        )

        print(
            f"Classical probability:      "
            f"{latest['Market_Probability']:.2f}%"
        )

        print(
            f"Quantum probability:       "
            f"{latest['Quantum_Score']:.2f}%"
        )

        print(
            f"Quantum hybrid score:      "
            f"{latest['Quantum_Hybrid_Score']:.2f}"
        )

        print(
            f"Quantum signal:             "
            f"{latest['Quantum_Signal']}"
        )

        print(
            f"Quantum hybrid signal:      "
            f"{latest['Quantum_Hybrid_Signal']}"
        )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()