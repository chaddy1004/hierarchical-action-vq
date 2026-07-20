"""
havq/analysis/factor.py
=======================
Scene-factoring test (diagnostics). The oracle probe (havq/analysis/probe.py)
showed v_t encodes the kitchen (participant) near-perfectly and only weakly, and
scene-entangledly, encodes the action. This asks: is the scene a LOW-RANK
nuisance we can just project out (making factoring easy), or is it pervasive?

Method: the between-participant subspace -- span of (participant-mean - global-
mean), <= n_participants-1 dims -- is the first-order "scene" direction. Project
it out of every v_t, then re-probe. If linear participant decodability collapses
toward chance while coarse action (category) decodability survives, scene is
low-rank-removable and the user's "factor out the scene" plan is cheap and
effective. If participant stays decodable, scene is high-rank / nonlinear and
needs stronger disentanglement.

Reuses load_dataset / make_splits / train_probe / pick_device from probe.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from havq.analysis.probe import load_dataset, make_splits, train_probe, pick_device
from havq.utils.config import load_config

logger = logging.getLogger(__name__)


def scene_basis(X: torch.Tensor, part: torch.Tensor, n_parts: int) -> torch.Tensor:
    """Orthonormal basis (r, D) of the between-participant mean subspace."""
    mu = X.mean(0, keepdim=True)
    means = torch.stack([X[part == p].mean(0) for p in range(n_parts)])  # (n_parts, D)
    M = means - mu
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = int((S > 1e-3 * S[0]).sum())
    logger.info(f"scene subspace rank kept: {r} (of <= {n_parts})")
    return Vh[:r]  # (r, D), orthonormal rows


def project_out(X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Remove the subspace spanned by Q (orthonormal rows) from every row of X."""
    return X - (X @ Q.t()) @ Q


def _probe_row(tag, X, y, splits, split_name, n_classes, cfg, device):
    tr, te = splits[split_name]
    m = {}
    for kind, hidden in (("linear", None), ("mlp", cfg["probe"]["mlp_hidden"])):
        r = train_probe(X, y, tr, te, n_classes, hidden, cfg, device)
        m[kind] = r
        logger.info(f"{tag:>28} {kind:<6} top1 {r['top1']:.3f} macroF1 {r['macro_f1']:.3f} "
                    f"(majority {r['majority_top1']:.3f}, chance {r['chance_top1']:.4f})")
    return m


def run(cfg: dict) -> dict:
    torch.manual_seed(cfg["probe"]["seed"])
    import numpy as np
    rng = np.random.default_rng(cfg["probe"]["seed"])
    device = pick_device(cfg["probe"]["device"])
    logger.info(f"device: {device}")

    X, verb, cat, part, vid, participants, cat_names = load_dataset(cfg)
    n_cats, n_parts = len(cat_names), len(participants)
    splits = make_splits(cfg, part, vid, participants, rng)

    Q = scene_basis(X, part, n_parts).contiguous()
    X_proj = project_out(X, Q)

    out = {"raw": {}, "scene_removed": {}, "scene_rank": Q.shape[0]}
    for name, Xv in (("raw", X), ("scene_removed", X_proj)):
        logger.info(f"===== {name} =====")
        out[name]["participant_random_clip"] = _probe_row(
            f"[{name}] PART random_clip", Xv, part, splits, "random_clip", n_parts, cfg, device)
        out[name]["category_heldout_video"] = _probe_row(
            f"[{name}] CAT heldout_video", Xv, cat, splits, "heldout_video", n_cats, cfg, device)
        out[name]["category_heldout_participant"] = _probe_row(
            f"[{name}] CAT heldout_part", Xv, cat, splits, "heldout_participant", n_cats, cfg, device)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "factor_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote {out_path}")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Scene-factoring test: project out the between-kitchen subspace, re-probe")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
