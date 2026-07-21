"""
run_visualize.py
================
Entry point for the LAPA latent-action HTML view (havq/visualization/visualize_lapa.py).

    python run_visualize.py --config havq/visualization/configs/visualize_h30.yaml
"""

import argparse
import logging

from havq.utils.config import load_config
from havq.visualization.visualize_lapa import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Render LAPA action labels to HTML")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
