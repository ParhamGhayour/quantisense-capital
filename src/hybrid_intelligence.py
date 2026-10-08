"""
Quantisense Capital
Hybrid Intelligence Engine - V1

Combines:
    1. Classical market intelligence
    2. Price-derived EMI
    3. Media intelligence

The system explicitly tracks whether real media data
is available for each observation.

No future market information is used in constructing
the hybrid score.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MARKET_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_quantisense_oos.csv"
)

MEDIA_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_media_interpretation.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_hybrid_intelligence.csv"
)


# ==========================================================
# HELPERS
# ==========================================================

def normalize_probability(series):
    """
    Convert probability 0-1 into a 0-100 scale.
    """

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .clip(0, 1)
        * 100.0
    )


# ==========================================================
# HYBRID ENGINE
# ==========================================================

def build_hybrid(market, media):

    market = market.copy()
    media = media.copy()

    # ------------------------------------------------------
    # Dates
    # ------------------------------------------------------

    market["Date"] = pd.to_datetime(
        market["Date"],
        errors="coerce"
    )

    media["Date"] = pd.to_datetime(
        media["Date"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Select media fields
    # ------------------------------------------------------

    media_columns = [
        "Date",
        "Media_Score",
        "Media_Fear",
        "Media_Uncertainty",
        "Media_Crisis",
        "Media_Risk",
        "Media_Volume",
        "Media_Volume_Shock",
        "Media_Narrative_Momentum",
        "Media_Confidence",
        "Media_Regime",
    ]

    media = media[
        [
            column
            for column in media_columns
            if column in media.columns
        ]
    ]

    # ------------------------------------------------------
    # Merge market + media
    #
    # Media at day t is used for the market outcome
    # following day t.
    # ------------------------------------------------------

    df = pd.merge(
        market,
        media,
        on="Date",
        how="left",
        suffixes=("", "_MEDIA"),
    )

    # ------------------------------------------------------
    # Determine whether real media exists
    # ------------------------------------------------------

    df["Media_Available"] = (
        df["Media_Score"].notna()
    )

    # ======================================================
    # MARKET INTELLIGENCE
    # ======================================================

    if "RF_Probability_Up" in df.columns:

        df["Market_Probability"] = (
            normalize_probability(
                df["RF_Probability_Up"]
            )
        )

    elif "RF_Confidence" in df.columns:

        confidence = pd.to_numeric(
            df["RF_Confidence"],
            errors="coerce"
        ).fillna(0)

        if "RF_Prediction" in df.columns:

            prediction = pd.to_numeric(
                df["RF_Prediction"],
                errors="coerce"
            ).fillna(0)

        else:

            prediction = pd.Series(
                0,
                index=df.index
            )

        df["Market_Probability"] = np.where(
            prediction == 1,
            50.0 + confidence / 2.0,
            50.0 - confidence / 2.0,
        )

    else:

        df["Market_Probability"] = 50.0

    df["Market_Probability"] = (
        pd.to_numeric(
            df["Market_Probability"],
            errors="coerce"
        )
        .fillna(50.0)
        .clip(0, 100)
    )

    # ======================================================
    # PRICE-DERIVED EMI
    # ======================================================

    if "EMI" in df.columns:

        df["Price_EMI"] = pd.to_numeric(
            df["EMI"],
            errors="coerce"
        ).clip(0, 100)

    elif "Quantisense_Score" in df.columns:

        df["Price_EMI"] = pd.to_numeric(
            df["Quantisense_Score"],
            errors="coerce"
        ).clip(0, 100)

    else:

        df["Price_EMI"] = 50.0

    df["Price_EMI"] = (
        df["Price_EMI"]
        .fillna(50.0)
        .clip(0, 100)
    )

    # ======================================================
    # MEDIA INTELLIGENCE
    # ======================================================

    df["Media_Score"] = pd.to_numeric(
        df["Media_Score"],
        errors="coerce"
    ).clip(0, 100)

    df["Media_Confidence"] = pd.to_numeric(
        df["Media_Confidence"],
        errors="coerce"
    ).clip(0, 100)

    # Keep actual media availability BEFORE filling NaN.

    df["Media_Available"] = (
        df["Media_Score"].notna()
    )

    # Missing media is treated as neutral for score
    # construction, but remains explicitly marked
    # as unavailable.

    df["Media_Score"] = (
        df["Media_Score"]
        .fillna(50.0)
    )

    df["Media_Confidence"] = (
        df["Media_Confidence"]
        .fillna(0.0)
    )

    # ======================================================
    # HYBRID WEIGHTS
    #
    # Base:
    #
    # Market intelligence       50%
    # Price EMI                 25%
    # Media intelligence        25%
    #
    # If media exists, its contribution depends on
    # media confidence.
    #
    # If media does NOT exist:
    # media contribution = 0
    # remaining weight goes to market intelligence.
    # ======================================================

    base_media_weight = (
        0.25
        * (
            0.50
            + 0.50
            * df["Media_Confidence"]
            / 100.0
        )
    )

    actual_media_weight = np.where(
        df["Media_Available"],
        base_media_weight,
        0.0,
    )

    df["Media_Weight_Actual"] = (
        actual_media_weight
    )

    price_emi_weight = 0.25

    market_weight = (
        1.0
        - price_emi_weight
        - actual_media_weight
    )

    df["Market_Weight_Actual"] = (
        market_weight
    )

    df["Price_EMI_Weight_Actual"] = (
        price_emi_weight
    )

    # ======================================================
    # HYBRID SCORE
    # ======================================================

    df["Hybrid_Score"] = (
        market_weight
        * df["Market_Probability"]

        + price_emi_weight
        * df["Price_EMI"]

        + actual_media_weight
        * df["Media_Score"]
    ).clip(0, 100)

    # ======================================================
    # HYBRID PROBABILITY
    # ======================================================

    df["Hybrid_Probability_Up"] = (
        df["Hybrid_Score"]
        / 100.0
    )

    # ======================================================
    # HYBRID SIGNAL
    # ======================================================

    df["Hybrid_Signal"] = np.select(
        [
            df["Hybrid_Score"] >= 70,

            df["Hybrid_Score"] >= 55,

            df["Hybrid_Score"] <= 30,

            df["Hybrid_Score"] <= 45,
        ],
        [
            "STRONG BUY",
            "BUY",
            "STRONG SELL",
            "SELL",
        ],
        default="HOLD",
    )

    # ======================================================
    # MARKET / MEDIA DIVERGENCE
    # ======================================================

    df["Market_Media_Divergence"] = (
        df["Market_Probability"]
        - df["Media_Score"]
    )

    df["Market_Media_Agreement"] = (
        100.0
        - df["Market_Media_Divergence"].abs()
    ).clip(0, 100)

    # ======================================================
    # INTERPRETATION
    # ======================================================

    df["Hybrid_Interpretation"] = np.select(
        [
            (
                df["Media_Available"]
                & (df["Market_Probability"] >= 60)
                & (df["Media_Score"] >= 60)
            ),

            (
                df["Media_Available"]
                & (df["Market_Probability"] <= 40)
                & (df["Media_Score"] <= 40)
            ),

            (
                df["Media_Available"]
                & (
                    df["Market_Media_Divergence"].abs()
                    >= 30
                )
            ),

            ~df["Media_Available"],
        ],
        [
            "Market and media confirm bullish conditions",

            "Market and media confirm bearish conditions",

            "Market and media disagree materially",

            "Market-only intelligence; media unavailable",
        ],
        default="Mixed or neutral conditions",
    )

    # ======================================================
    # DATA QUALITY
    # ======================================================

    df["Hybrid_Data_Quality"] = np.select(
        [
            (
                df["Media_Available"]
                & (df["Media_Confidence"] >= 60)
            ),

            df["Media_Available"],

            ~df["Media_Available"],
        ],
        [
            "FULL_MEDIA",

            "MEDIA_AVAILABLE_LOW_CONFIDENCE",

            "MARKET_ONLY",
        ],
        default="UNKNOWN",
    )

    return df


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("QUANTISENSE HYBRID INTELLIGENCE")
    print("=" * 70)

    # ------------------------------------------------------
    # Market
    # ------------------------------------------------------

    print()
    print("Loading market intelligence...")

    if not MARKET_PATH.exists():

        raise FileNotFoundError(
            f"Market file not found:\n{MARKET_PATH}"
        )

    market = pd.read_csv(
        MARKET_PATH
    )

    print(
        f"Market rows: {len(market)}"
    )

    # ------------------------------------------------------
    # Media
    # ------------------------------------------------------

    print()
    print("Loading media intelligence...")

    if not MEDIA_PATH.exists():

        raise FileNotFoundError(
            f"Media file not found:\n{MEDIA_PATH}"
        )

    media = pd.read_csv(
        MEDIA_PATH
    )

    print(
        f"Media rows: {len(media)}"
    )

    # ------------------------------------------------------
    # Build hybrid
    # ------------------------------------------------------

    df = build_hybrid(
        market,
        media
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    media_available_count = int(
        df["Media_Available"].sum()
    )

    market_only_count = int(
        (~df["Media_Available"]).sum()
    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()
    print(
        f"Hybrid rows: {len(df)}"
    )

    print(
        f"Rows with real media: "
        f"{media_available_count}"
    )

    print(
        f"Market-only rows: "
        f"{market_only_count}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    # ------------------------------------------------------
    # Latest state
    # ------------------------------------------------------

    if len(df) > 0:

        latest = df.iloc[-1]

        print()
        print("LATEST HYBRID STATE")
        print("-" * 70)

        print(
            f"Date:                    "
            f"{latest['Date']}"
        )

        print(
            f"Market Probability:      "
            f"{latest['Market_Probability']:.2f}"
        )

        print(
            f"Price EMI:               "
            f"{latest['Price_EMI']:.2f}"
        )

        print(
            f"Media Available:         "
            f"{latest['Media_Available']}"
        )

        print(
            f"Media Score:             "
            f"{latest['Media_Score']:.2f}"
        )

        print(
            f"Media Confidence:        "
            f"{latest['Media_Confidence']:.2f}"
        )

        print(
            f"Hybrid Score:            "
            f"{latest['Hybrid_Score']:.2f}"
        )

        print(
            f"Hybrid Probability:      "
            f"{latest['Hybrid_Probability_Up']:.2%}"
        )

        print(
            f"Hybrid Signal:           "
            f"{latest['Hybrid_Signal']}"
        )

        print(
            f"Market Weight:           "
            f"{latest['Market_Weight_Actual']:.2%}"
        )

        print(
            f"Price EMI Weight:        "
            f"{latest['Price_EMI_Weight_Actual']:.2%}"
        )

        print(
            f"Media Weight:            "
            f"{latest['Media_Weight_Actual']:.2%}"
        )

        print(
            f"Market/Media Agreement:  "
            f"{latest['Market_Media_Agreement']:.2f}"
        )

        print(
            f"Data Quality:             "
            f"{latest['Hybrid_Data_Quality']}"
        )

        print(
            f"Interpretation:          "
            f"{latest['Hybrid_Interpretation']}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()