"""
havq/tokenize_videos.py
=======================
Apply the frozen trained NSVQ model (havq/train.py output) to every cached
feature file, producing one discrete latent-action token per (v_t, v_{t+H}) pair
via hard nearest-code assignment (no noise).

Per video (n_clips clips -> n_clips - H tokens):

    <tokens_dir>/<video_id>.npy         int32 (n_clips - H,)  token stream
    <tokens_dir>/<video_id>_tokens.json run-length / usage stats for a quick look

Token index j corresponds to the transition from clip j to clip j+H, so a token
*change* between j-1 and j marks a predicted action boundary near clip j — the
signal the §9 analysis will compare against the GT boundaries from havq/gt.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
import torch

from havq import load_config
from havq.vq import NSVQ, l2_normalize

logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, device: str) -> tuple[NSVQ, int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = NSVQ(**ckpt["hparams"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["H"]


def token_stats(tokens: np.ndarray, codebook_size: int) -> dict:
    """Run-length summary for a quick first look at a token stream."""
    changes = int((tokens[1:] != tokens[:-1]).sum()) if len(tokens) > 1 else 0
    # run lengths of consecutive identical tokens
    if len(tokens) == 0:
        run_lengths = []
    else:
        boundaries = np.flatnonzero(np.diff(tokens)) + 1
        run_lengths = np.diff([0, *boundaries.tolist(), len(tokens)]).tolist()
    return {
        "n_tokens": int(len(tokens)),
        "n_unique": int(len(np.unique(tokens))),
        "codebook_size": codebook_size,
        "n_changes": changes,
        "mean_run_length": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "max_run_length": int(np.max(run_lengths)) if run_lengths else 0,
    }


def tokenize_all(cfg: dict) -> None:
    features_dir = cfg["paths"]["features_dir"]
    tokens_dir = cfg["paths"]["tokens_dir"]
    device = cfg["train"]["device"]
    os.makedirs(tokens_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg["paths"]["results_dir"], "nsvq.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found — run havq/train.py first")
    model, H = load_model(ckpt_path, device)

    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files in {features_dir}")

    for f in npy_files:
        video_id = os.path.splitext(os.path.basename(f))[0]
        V = l2_normalize(np.load(f).astype(np.float32))
        if V.shape[0] <= H:
            logger.warning(f"{video_id}: only {V.shape[0]} clips (<= H={H}), skipping")
            continue

        vt = torch.from_numpy(V[:-H]).to(device)
        vtH = torch.from_numpy(V[H:]).to(device)
        tokens = model.encode_indices(vt, vtH).cpu().numpy().astype(np.int32)

        np.save(os.path.join(tokens_dir, video_id + ".npy"), tokens)
        stats = token_stats(tokens, model.codebook_size)
        with open(os.path.join(tokens_dir, video_id + "_tokens.json"), "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info(
            f"{video_id}: {stats['n_tokens']} tokens, {stats['n_unique']} unique, "
            f"{stats['n_changes']} changes, mean run {stats['mean_run_length']:.1f} clips"
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Tokenize cached V-JEPA features with the trained NSVQ model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    tokenize_all(load_config(args.config))


if __name__ == "__main__":
    main()
