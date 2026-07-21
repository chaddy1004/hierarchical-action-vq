"""
havq/utils/config.py
====================
Config loading. The config is a plain nested dict loaded from YAML — no Hydra
/ OmegaConf, access stays explicit (same convention as stepsegmenter).
"""

from __future__ import annotations

import os

import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
