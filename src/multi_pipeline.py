
"""
Quantisense Capital
Multi Asset Market Pipeline
"""

from src.config_loader import load_assets
from src.market_data import download_market_data
from src.validation import validate_market_data
from src.features import add_daily_return
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

        validation = validate_market_data(data)

        data = add_daily_return(data)

        output_path = f"{output_dir}/{asset.replace('-', '_')}_features.csv"

        save_dataset(
            data,
            output_path
        )

        results[asset] = {
            "validation": validation,
            "output": output_path
        }

    return results
