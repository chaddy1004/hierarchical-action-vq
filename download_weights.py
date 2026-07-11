"""Download backbone weights to an explicit directory (not the hidden HF cache).

Covers the V-JEPA2 encoder family -- all drop-in swappable with the existing
VJepaExtractor (havq/data/preprocessing/extractors/vjepa_extractor.py), just by
pointing a preprocessing config's `model_weight` at a different directory here.
vjepa2-vitg-fpc64-384 is the one currently used (see configs/*.yaml).

Not included: V-JEPA2-AC (action-conditioned variant) and LaVILA, both named as
possible future backbones in RESEARCH.md. Neither is a drop-in swap for the
current extractor -- AC has different conditioning inputs/outputs, and LaVILA
is a dual text-video encoder, not a plain video encoder -- so downloading them
now would just be dead weight until an extractor class exists for them.

Usage (run on a login node -- compute nodes have no internet):
    python download_weights.py --dest /path/to/scratch/models                    # all four
    python download_weights.py --dest /path/to/scratch/models --models vjepa2-vitg-fpc64-384
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

# name -> HF repo id. All ungated / apache-2.0, no token needed.
WEIGHTS = {
    "vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2-vith-fpc64-256": "facebook/vjepa2-vith-fpc64-256",
    "vjepa2-vitg-fpc64-256": "facebook/vjepa2-vitg-fpc64-256",
    "vjepa2-vitg-fpc64-384": "facebook/vjepa2-vitg-fpc64-384",  # current default backbone
}

# safetensors only -- skip redundant .bin/.h5/.msgpack/.onnx copies of the same weights.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


def main():
    ap = argparse.ArgumentParser(description="Download backbone weights to an explicit directory")
    ap.add_argument("--models", default=",".join(WEIGHTS.keys()),
                     help=f"comma-separated names from {list(WEIGHTS.keys())}")
    ap.add_argument("--dest", required=True, help="destination root directory (explicit -- no default, "
                     "so you always know exactly where these land, e.g. a real /scratch path, not /home)")
    args = ap.parse_args()

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in WEIGHTS]
    if unknown:
        raise ValueError(f"Unknown model name(s) {unknown}. Choices: {list(WEIGHTS.keys())}")

    os.makedirs(args.dest, exist_ok=True)
    for name in names:
        repo_id = WEIGHTS[name]
        local_dir = os.path.join(args.dest, name)
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=ALLOW_PATTERNS)
        print(f"  done: {local_dir}")


if __name__ == "__main__":
    main()
