"""
run_gate.py
===========
Entry point for the hicap Experiment 1 gate (hicap/eval/gate.py).

    python run_gate.py --config hicap/configs/gate_50salads.yaml
    python run_gate.py --config ... --max-videos 5     # quick smoke run
"""

import argparse
import logging

from hicap.eval.gate import run
from hicap.utils.config import load_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="hicap count-free boundary gate on 50 Salads")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-videos", type=int, default=None, help="override: only first N videos")
    parser.add_argument("--salads-root", default=None, help="override: parent dir of 50salads/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_videos is not None:
        cfg["max_videos"] = args.max_videos
    if args.salads_root is not None:
        cfg["paths"]["salads_root"] = args.salads_root
    run(cfg)


if __name__ == "__main__":
    main()
