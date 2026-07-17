
"""
Quantisense Capital
Configuration Loader
"""

import yaml


def load_assets(path="config/assets.yaml"):
    """
    Load asset list from YAML configuration.
    """

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config["assets"]
