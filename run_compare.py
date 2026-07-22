"""
run_compare.py
==============
Entry point for the known-K unsupervised-TAS comparison (hicap/eval/compare.py).

    python run_compare.py --config hicap/configs/compare_50salads.yaml
    python run_compare.py --config ... --max-videos 5
"""

import argparse
import logging

from hicap.eval.compare import run
from hicap.utils.config import load_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="hicap known-K TAS comparison on 50 Salads")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="override: 50salads | breakfast | gtea")
    parser.add_argument("--data-root", default=None, help="override: parent dir of <dataset>/")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_videos is not None:
        cfg["max_videos"] = args.max_videos
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.data_root is not None:
        cfg["paths"]["data_root"] = args.data_root
    run(cfg)


if __name__ == "__main__":
    main()
