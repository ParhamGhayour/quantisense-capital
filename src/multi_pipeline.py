"""
Quantisense Capital
Integrated Multi-Asset Phase 1 Pipeline
"""

import pandas as pd

from src.config_loader import load_assets
from src.market_data import download_market_data
from src.validation import validate_market_data
from src.features import add_daily_return
from src.indicators import add_returns, add_volatility, add_ema_indicators
from src.atr import add_true_range, add_atr
from src.signals import add_ema_score, add_volatility_regime
from src.scoring import add_market_score
from src.storage import save_dataset


def run_multi_asset_pipeline(
    start,
    end,
    output_dir="data"
):

    assets = load_assets()
    results = {}

    for asset in assets:

        data = download_market_data(
            asset,
            start,
            end
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(how="all")

        validation = validate_market_data(data)

        data = add_daily_return(data)
        data = add_ema_indicators(data)
        data = add_returns(data)
        data = add_volatility(data)
        data = add_true_range(data)
        data = add_atr(data)
        data = add_ema_score(data)
        data = add_volatility_regime(data)
        data = add_market_score(data)

        data = data.dropna(how="any").reset_index(drop=True)

        output_path = (
            f"{output_dir}/{asset.replace('-', '_')}_phase1.csv"
        )

        save_dataset(
            data,
            output_path
        )

        results[asset] = {
            "validation": validation,
            "output": output_path
        }

    return results