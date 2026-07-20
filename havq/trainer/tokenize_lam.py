"""
havq/trainer/tokenize_lam.py
============================
Apply the frozen trained NSVQ model (havq/trainer/train_lam.py output) to every
cached feature file: one discrete latent-action token per (v_t, v_{t+H}) pair via
hard nearest-code assignment (no noise).

Per video (n_clips clips -> n_clips - H tokens):
    <tokens_dir>/<id>.npy         int32 (n_clips - H,)  token stream
    <tokens_dir>/<id>_tokens.json run-length / usage stats

Token j = the transition clip j -> j+H; a token *change* between j-1 and j marks
a predicted action boundary near clip j (consumed by havq/analysis/bpe_eval.py).
Adapted from archive/v1-derisk/havq/tokenize_videos.py (new cache layout).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
import torch

from havq.model.vq import NSVQ, l2_normalize
from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = NSVQ(**ckpt["hparams"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["H"]


def token_stats(tokens: np.ndarray, codebook_size: int) -> dict:
    changes = int((tokens[1:] != tokens[:-1]).sum()) if len(tokens) > 1 else 0
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


def pick_device(pref: str) -> str:
    return "cuda" if (pref == "cuda" and torch.cuda.is_available()) else "cpu"


def tokenize_all(cfg: dict) -> None:
    features_dir = os.path.join(
        cfg["paths"]["features_root"],
        feature_subdir(cfg["backbone"], cfg["scale"]["n_clip_frame"], cfg["scale"]["stride"]),
    )
    tokens_dir = cfg["paths"]["tokens_dir"]
    device = pick_device(cfg["train"]["device"])
    os.makedirs(tokens_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg["paths"]["results_dir"], "nsvq.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found -- run havq.trainer.train_lam first")
    model, H = load_model(ckpt_path, device)

    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files in {features_dir}")

    mean_runs = []
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
        mean_runs.append(stats["mean_run_length"])

    logger.info(f"Tokenized {len(mean_runs)} videos -> {tokens_dir}. "
                f"Mean run length across videos: {np.mean(mean_runs):.2f} clips "
                f"(1.0 = flickering every step).")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Tokenize cached features with the trained NSVQ model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    tokenize_all(load_config(args.config))


if __name__ == "__main__":
    main()
