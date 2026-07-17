
"""
Quantisense Capital
Market Data Pipeline
"""

from src.market_data import download_market_data
from src.validation import validate_market_data
from src.features import add_daily_return
from src.storage import save_dataset


def run_market_pipeline(
    ticker,
    start,
    end,
    output_path
):

    data = download_market_data(
        ticker,
        start,
        end
    )

    validation = validate_market_data(data)

    data = add_daily_return(data)

    save_dataset(
        data,
        output_path
    )

    return {
        "validation": validation,
        "output": output_path
    }
