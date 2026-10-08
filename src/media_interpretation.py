"""
Quantisense Capital
Media Interpretation Engine - V1

Transforms GDELT-derived daily media statistics into
interpretable, deterministic media intelligence.

This is NOT an LLM-based interpretation layer.
It is reproducible quantitative media analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "gdelt_btc_daily_test.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "BTC_USD_media_interpretation.csv"
)


def causal_minmax(series, low=0.0, high=100.0):
    """
    Normalize using historical information only.

    The current observation is NOT allowed to determine
    its own normalization range.
    """

    historical_min = series.expanding().min().shift(1)
    historical_max = series.expanding().max().shift(1)

    denominator = (
        historical_max - historical_min
    ).replace(0, np.nan)

    result = (
        (series - historical_min)
        / denominator
    )

    result = (
        result
        .clip(0, 1)
        * (high - low)
        + low
    )

    return result.fillna(50.0)


def build_media_interpretation(df):

    df = df.copy()

    # --------------------------------------------------
    # Date
    # --------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    # --------------------------------------------------
    # Numeric media fields
    # --------------------------------------------------

    numeric_columns = [
        "Bitcoin_Records",
        "Bitcoin_Articles",
        "Bitcoin_Tone_Mean",
        "Bitcoin_Tone_Median",
        "Bitcoin_Negative_Share",
        "Bitcoin_Positive_Share",
        "Bitcoin_Neutral_Share",
        "Bitcoin_Uncertainty_Share",
        "Bitcoin_Policy_Share",
        "Bitcoin_Economy_Share",
        "Bitcoin_Crisis_Share",
        "Bitcoin_Financial_Risk_Share",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------
    # 1. MEDIA SENTIMENT
    #
    # Tone is approximately centered around zero.
    # Convert it to a 0-100 interpretation scale.
    # --------------------------------------------------

    df["Media_Sentiment"] = (
        50.0
        + df["Bitcoin_Tone_Mean"] * 5.0
    ).clip(0, 100)

    # --------------------------------------------------
    # 2. MEDIA FEAR
    #
    # Negative tone
    # + uncertainty
    # + crisis
    # + financial risk
    # --------------------------------------------------

    df["Media_Fear"] = (
        100.0
        * (
            0.40
            * df["Bitcoin_Negative_Share"]

            + 0.25
            * df["Bitcoin_Uncertainty_Share"]

            + 0.20
            * df["Bitcoin_Crisis_Share"]

            + 0.15
            * df["Bitcoin_Financial_Risk_Share"]
        )
    ).clip(0, 100)

    # --------------------------------------------------
    # 3. MEDIA UNCERTAINTY
    # --------------------------------------------------

    df["Media_Uncertainty"] = (
        100.0
        * df["Bitcoin_Uncertainty_Share"]
    ).clip(0, 100)

    # --------------------------------------------------
    # 4. MEDIA CRISIS
    # --------------------------------------------------

    df["Media_Crisis"] = (
        100.0
        * df["Bitcoin_Crisis_Share"]
    ).clip(0, 100)

    # --------------------------------------------------
    # 5. MEDIA FINANCIAL RISK
    # --------------------------------------------------

    df["Media_Risk"] = (
        100.0
        * (
            0.45
            * df["Bitcoin_Financial_Risk_Share"]

            + 0.30
            * df["Bitcoin_Crisis_Share"]

            + 0.25
            * df["Bitcoin_Uncertainty_Share"]
        )
    ).clip(0, 100)

    # --------------------------------------------------
    # 6. MEDIA VOLUME
    # --------------------------------------------------

    df["Media_Volume"] = (
        df["Bitcoin_Articles"]
        .fillna(df["Bitcoin_Records"])
        .fillna(0)
    )

    # --------------------------------------------------
    # 7. MEDIA VOLUME SHOCK
    # --------------------------------------------------

    volume_average = (
        df["Media_Volume"]
        .rolling(7)
        .mean()
    )

    df["Media_Volume_Shock"] = (
        df["Media_Volume"]
        / volume_average
    )

    df["Media_Volume_Shock"] = (
        df["Media_Volume_Shock"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(1.0)
    )

    # --------------------------------------------------
    # 8. NARRATIVE MOMENTUM
    #
    # Positive = improving media narrative
    # Negative = deteriorating media narrative
    # --------------------------------------------------

    df["Media_Tone_Change_1D"] = (
        df["Bitcoin_Tone_Mean"].diff()
    )

    df["Media_Tone_Change_3D"] = (
        df["Bitcoin_Tone_Mean"].diff(3)
    )

    df["Media_Tone_Change_7D"] = (
        df["Bitcoin_Tone_Mean"].diff(7)
    )

    narrative_normalized = causal_minmax(
        df["Media_Tone_Change_3D"].fillna(0)
    )

    df["Media_Narrative_Momentum"] = (
        50.0
        + narrative_normalized / 2.0
    ).clip(0, 100)

    # --------------------------------------------------
    # 9. COMBINED MEDIA SCORE
    #
    # Sentiment       35%
    # Fear            25%
    # Uncertainty     15%
    # Crisis          10%
    # Narrative       15%
    # --------------------------------------------------

    df["Media_Score"] = (
        0.35 * df["Media_Sentiment"]

        + 0.25
        * (100.0 - df["Media_Fear"])

        + 0.15
        * (100.0 - df["Media_Uncertainty"])

        + 0.10
        * (100.0 - df["Media_Crisis"])

        + 0.15
        * df["Media_Narrative_Momentum"]
    ).clip(0, 100)

    # --------------------------------------------------
    # 10. MEDIA REGIME
    # --------------------------------------------------

    df["Media_Regime"] = np.select(
        [
            df["Media_Score"] >= 75,

            df["Media_Score"] >= 60,

            df["Media_Score"] <= 25,

            df["Media_Score"] <= 40,
        ],
        [
            "Extreme Optimism",

            "Optimism",

            "Extreme Fear",

            "Fear",
        ],
        default="Neutral",
    )

    # --------------------------------------------------
    # 11. MEDIA CONFIDENCE
    #
    # More media coverage +
    # stronger narrative movement =
    # greater confidence.
    # --------------------------------------------------

    volume_confidence = causal_minmax(
        df["Media_Volume"]
    )

    narrative_strength = (
        df["Media_Tone_Change_3D"]
        .abs()
        .fillna(0)
    )

    narrative_confidence = causal_minmax(
        narrative_strength
    )

    df["Media_Confidence"] = (
        0.60 * volume_confidence
        + 0.40 * narrative_confidence
    ).clip(0, 100)

    # --------------------------------------------------
    # 12. SAVE
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("=" * 70)
    print("QUANTISENSE MEDIA INTERPRETATION")
    print("=" * 70)

    print()
    print(f"Rows: {len(df)}")

    print()
    print(f"Output: {OUTPUT_PATH}")

    if len(df) > 0:

        latest = df.iloc[-1]

        print()
        print("LATEST MEDIA INTERPRETATION")
        print("-" * 70)

        print(
            f"Date:                 {latest['Date']}"
        )

        print(
            f"Media Score:          "
            f"{latest['Media_Score']:.2f}"
        )

        print(
            f"Media Regime:         "
            f"{latest['Media_Regime']}"
        )

        print(
            f"Media Sentiment:      "
            f"{latest['Media_Sentiment']:.2f}"
        )

        print(
            f"Media Fear:           "
            f"{latest['Media_Fear']:.2f}"
        )

        print(
            f"Media Uncertainty:    "
            f"{latest['Media_Uncertainty']:.2f}"
        )

        print(
            f"Media Crisis:         "
            f"{latest['Media_Crisis']:.2f}"
        )

        print(
            f"Media Risk:           "
            f"{latest['Media_Risk']:.2f}"
        )

        print(
            f"Media Volume:         "
            f"{latest['Media_Volume']:.0f}"
        )

        print(
            f"Media Confidence:     "
            f"{latest['Media_Confidence']:.2f}"
        )

    print()
    print("=" * 70)


def main():

    print("Loading media data...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows loaded: {len(df)}")

    result = build_media_interpretation(df)

    print()
    print("Media interpretation completed successfully.")


if __name__ == "__main__":
    main()