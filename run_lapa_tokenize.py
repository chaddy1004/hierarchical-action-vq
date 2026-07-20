"""
run_lapa_tokenize.py
====================
Entry point for LAPA latent-action tokenization (havq/data/preprocessing/lapa_tokenize.py).

    python run_lapa_tokenize.py --config havq/data/preprocessing/configs/lapa_h30.yaml
    python run_lapa_tokenize.py --config ... --video-ids P05-20240424-090812
"""

import argparse
import logging

from havq.data.preprocessing.lapa_tokenize import tokenize_all
from havq.utils.config import load_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Tokenize videos into LAPA latent actions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="re-tokenize cached videos")
    parser.add_argument("--video-ids", nargs="+", default=None, help="only these video ids")
    args = parser.parse_args()

    out_dir = tokenize_all(load_config(args.config), overwrite=args.overwrite, video_ids=args.video_ids)
    logging.getLogger(__name__).info(f"Wrote latent actions to {out_dir}")


if __name__ == "__main__":
    main()
