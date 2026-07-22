"""
run_vjepa_extract.py
====================
Entry point for V-JEPA feature extraction from RGB (hicap/data/vjepa_extract.py).

    python run_vjepa_extract.py --config hicap/configs/vjepa_breakfast.yaml

Runs on a GPU node. On Compute Canada, submit via hicap/slurm/extract_vjepa.slurm.
"""

import argparse
import logging

from hicap.data.vjepa_extract import extract_all
from hicap.utils.config import load_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Extract V-JEPA features from RGB videos")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.overwrite:
        cfg["overwrite"] = True
    extract_all(cfg)


if __name__ == "__main__":
    main()
