"""
run_hierarchy.py
================
Entry point for single-video bottom-up hierarchical segmentation
(havq/analysis/hierarchy.py).

    python run_hierarchy.py --config havq/analysis/configs/hierarchy_h15.yaml
"""

import argparse
import logging

from havq.analysis.hierarchy import run
from havq.utils.config import load_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build and score a per-video action hierarchy")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
