"""
havq/analysis/temporal_split.py
===============================
De-risk test for the two-codebook idea (PROBE_REPORT.md §7): does the crude,
learning-free split "scene = per-video mean, action = residual" actually
separate scene from action?

For each clip v_t in a video define:
    videomean : v_bar = mean over all clips of that video   (constant per video)
    residual  : r_t   = v_t - v_bar
Then re-probe three representations (raw v_t, videomean, residual) for category
(the action signal) and participant (the scene). If the residual sheds scene
(participant decodability drops vs raw) while keeping action (category holds
cross-kitchen), the "scene=slow, action=fast" inductive bias is real and worth
learning a better version of. If not, the bias is wrong.

Reuses load_dataset / make_splits / train_probe / pick_device from probe.py.
No VQ, no learned projection -- this is the linear stand-in.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from havq.analysis.probe import load_dataset, make_splits, train_probe, pick_device
from havq.utils.config import load_config

logger = logging.getLogger(__name__)


def video_mean_residual(X: torch.Tensor, vid: torch.Tensor):
    """Per-video mean (broadcast to each clip) and the residual v_t - mean."""
    n_vid = int(vid.max()) + 1
    videomean = torch.empty_like(X)
    for v in range(n_vid):
        m = vid == v
        if m.any():
            videomean[m] = X[m].mean(0, keepdim=True)
    return videomean, X - videomean


def run(cfg: dict) -> dict:
    torch.manual_seed(cfg["probe"]["seed"])
    rng = np.random.default_rng(cfg["probe"]["seed"])
    device = pick_device(cfg["probe"]["device"])
    logger.info(f"device: {device}")

    X, verb, cat, part, vid, participants, cat_names = load_dataset(cfg)
    n_cats, n_parts = len(cat_names), len(participants)
    splits = make_splits(cfg, part, vid, participants, rng)

    videomean, residual = video_mean_residual(X, vid)
    reps = {"raw": X, "videomean": videomean, "residual": residual}

    out = {}
    for rep_name, Xv in reps.items():
        out[rep_name] = {"category": {}, "participant": {}}
        # category (action) across all three splits
        for split_name, (tr, te) in splits.items():
            if len(te) == 0:
                continue
            row = {}
            for kind, hidden in (("linear", None), ("mlp", cfg["probe"]["mlp_hidden"])):
                m = train_probe(Xv, cat, tr, te, n_cats, hidden, cfg, device)
                row[kind] = m
                logger.info(f"CAT  [{rep_name:>9}][{split_name:>18}] {kind:<6} "
                            f"top1 {m['top1']:.3f} macroF1 {m['macro_f1']:.3f} "
                            f"(majority {m['majority_top1']:.3f})")
            out[rep_name]["category"][split_name] = row
        # participant (scene) on random_clip only -- labels must appear in train
        tr, te = splits["random_clip"]
        row = {}
        for kind, hidden in (("linear", None), ("mlp", cfg["probe"]["mlp_hidden"])):
            m = train_probe(Xv, part, tr, te, n_parts, hidden, cfg, device)
            row[kind] = m
            logger.info(f"PART [{rep_name:>9}][       random_clip] {kind:<6} "
                        f"top1 {m['top1']:.3f} (chance {m['chance_top1']:.4f})")
        out[rep_name]["participant"]["random_clip"] = row

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    p = os.path.join(results_dir, "temporal_split_report.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {p}")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Video-mean vs residual factoring de-risk test")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
