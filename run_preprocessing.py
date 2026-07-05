"""Entry point: cache V-JEPA clip features for every video in the config's
`paths.videos_dir`. All settings live in the config."""

import argparse
import logging
import os

from havq.data.preprocessing.preprocessing import Preprocessor
from havq.utils.config import load_config

# preprocessing has its own standalone config (real dataset paths + backbone
# weights), separate from the main training config.
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "havq", "data", "preprocessing", "configs", "vjepa_clip10.yaml"
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Clip feature extraction")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--overwrite", action="store_true", help="re-extract cached videos")
    args = ap.parse_args()

    preprocessor = Preprocessor(load_config(args.config))
    out_dir = preprocessor.feature_extraction(overwrite=args.overwrite)
    print(f"Features cached in {out_dir}")


if __name__ == "__main__":
    main()
