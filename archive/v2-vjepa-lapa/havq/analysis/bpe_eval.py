"""
havq/analysis/bpe_eval.py
=========================
Does BPE composition of the latent-action token stream recover a segmentation
that beats random at *some* granularity? Run on the HELD-OUT videos only (the
val set saved by havq/trainer/train_lam.py) so nothing here was trained on.

Pipeline per video:
  1. token stream -> RLE (collapse runs of identical tokens) = finest segmentation.
  2. BPE: repeatedly merge the most frequent adjacent symbol pair across the
     held-out corpus, recording the merge tree. Each merge removes the boundary
     between that pair's occurrences -> a coarser segmentation.
  3. At snapshot levels, convert the remaining boundaries (token index j -> time
     j / clips_per_sec) and score boundary-F1 vs the GT narration boundaries,
     against a matched-count random baseline (same #boundaries per video).

Reports per-level mean boundary-F1 (model vs random) across the held-out videos
and the best level. Boundary-F1 machinery copied from archive/v1-derisk/havq/eval.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter

import numpy as np

from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- boundary F1

def match_count(pred_sec: np.ndarray, gt_sec: np.ndarray, tol: float) -> int:
    if len(pred_sec) == 0 or len(gt_sec) == 0:
        return 0
    pairs = []
    for i, p in enumerate(pred_sec):
        for j, g in enumerate(gt_sec):
            d = abs(p - g)
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp


def f1(pred_sec: np.ndarray, gt_sec: np.ndarray, tol: float) -> float:
    tp = match_count(pred_sec, gt_sec, tol)
    prec = tp / len(pred_sec) if len(pred_sec) else 0.0
    rec = tp / len(gt_sec) if len(gt_sec) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def random_f1(n_pred: int, n_positions: int, gt_sec: np.ndarray, clips_per_sec: float,
              tol: float, trials: int, rng) -> float:
    if n_pred == 0 or n_positions <= 1:
        return 0.0
    k = min(n_pred, n_positions - 1)
    return float(np.mean([
        f1(np.sort(rng.choice(np.arange(1, n_positions), size=k, replace=False)) / clips_per_sec, gt_sec, tol)
        for _ in range(trials)
    ]))


# ---------------------------------------------------------------- RLE + BPE

def rle(tokens: np.ndarray):
    """Return (symbols, starts): maximal runs of identical tokens. starts[k] is
    the token index where symbol k begins. Boundaries = starts[1:]."""
    if len(tokens) == 0:
        return [], []
    symbols, starts = [int(tokens[0])], [0]
    for i in range(1, len(tokens)):
        if tokens[i] != tokens[i - 1]:
            symbols.append(int(tokens[i])); starts.append(i)
    return symbols, starts


def bpe_snapshots(seqs, num_merges: int, levels: set):
    """seqs: list of [symbols, starts] (mutated). Snapshot each video's boundary
    token-indices (current starts[1:]) at merge counts in `levels`. Returns
    {level: [boundaries_per_video]}."""
    next_id = max((s for sy, st in seqs for s in sy), default=0) + 1
    snaps = {}
    if 0 in levels:
        snaps[0] = [list(st[1:]) for sy, st in seqs]
    for m in range(1, num_merges + 1):
        pairs = Counter()
        for sy, _ in seqs:
            for i in range(len(sy) - 1):
                pairs[(sy[i], sy[i + 1])] += 1
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:  # no adjacent pair recurs corpus-wide -> stop merging
            break
        new = next_id; next_id += 1
        for rec in seqs:
            sy, st = rec
            nsy, nst, i = [], [], 0
            while i < len(sy):
                if i < len(sy) - 1 and sy[i] == a and sy[i + 1] == b:
                    nsy.append(new); nst.append(st[i]); i += 2
                else:
                    nsy.append(sy[i]); nst.append(st[i]); i += 1
            rec[0], rec[1] = nsy, nst
        if m in levels:
            snaps[m] = [list(st[1:]) for sy, st in seqs]
    return snaps


# ---------------------------------------------------------------- driver

def run(cfg: dict) -> dict:
    ecfg = cfg["eval"]
    tol = ecfg["primary_tolerance_sec"]
    rng = np.random.default_rng(cfg["train"]["seed"])

    features_dir = os.path.join(
        cfg["paths"]["features_root"],
        feature_subdir(cfg["backbone"], cfg["scale"]["n_clip_frame"], cfg["scale"]["stride"]),
    )
    tokens_dir = cfg["paths"]["tokens_dir"]
    results_dir = cfg["paths"]["results_dir"]
    with open(os.path.join(results_dir, "train_log.json")) as f:
        val_ids = json.load(f)["val_video_ids"]
    logger.info(f"Evaluating BPE composition on {len(val_ids)} held-out videos (tol {tol}s)")

    videos, seqs = [], []
    for vid in val_ids:
        tok_path = os.path.join(tokens_dir, vid + ".npy")
        gt_path = os.path.join(features_dir, vid + "_gt.npz")
        meta_path = os.path.join(features_dir, vid + "_meta.json")
        if not (os.path.exists(tok_path) and os.path.exists(gt_path) and os.path.exists(meta_path)):
            logger.warning(f"{vid}: missing tokens/gt/meta, skipping")
            continue
        tokens = np.load(tok_path)
        gt_sec = np.load(gt_path)["boundaries_sec"]
        with open(meta_path) as f:
            meta = json.load(f)
        cps = meta["fps"] / meta["n_clip_frame"]
        if len(tokens) < 2 or len(gt_sec) == 0:
            continue
        symbols, starts = rle(tokens)
        videos.append({"vid": vid, "gt_sec": gt_sec, "cps": cps, "n_tokens": len(tokens)})
        seqs.append([symbols, starts])

    # snapshot levels: 0 (RLE) + geometric spacing up to num_merges
    num_merges = ecfg["num_merges"]
    levels = {0} | {int(round(x)) for x in np.geomspace(1, num_merges, ecfg["num_levels"])}
    snaps = bpe_snapshots(seqs, num_merges, levels)

    report = {"tol": tol, "n_videos": len(videos), "levels": []}
    for level in sorted(snaps):
        bset = snaps[level]
        model_f1s, rand_f1s, npreds = [], [], []
        for v, bounds in zip(videos, bset):
            pred_sec = np.array(sorted(bounds)) / v["cps"]
            model_f1s.append(f1(pred_sec, v["gt_sec"], tol))
            rand_f1s.append(random_f1(len(bounds), v["n_tokens"], v["gt_sec"], v["cps"],
                                      tol, ecfg["random_trials"], rng))
            npreds.append(len(bounds))
        report["levels"].append({
            "merges": level,
            "mean_model_f1": float(np.mean(model_f1s)),
            "mean_random_f1": float(np.mean(rand_f1s)),
            "mean_boundaries_per_video": float(np.mean(npreds)),
        })
        logger.info(f"merges {level:5d} | model F1 {np.mean(model_f1s):.3f} | "
                    f"random {np.mean(rand_f1s):.3f} | ~{np.mean(npreds):.0f} bnds/vid")

    best = max(report["levels"], key=lambda r: r["mean_model_f1"] - r["mean_random_f1"])
    report["best_level"] = best
    logger.info("=" * 60)
    logger.info(f"BEST: merges {best['merges']} | model F1 {best['mean_model_f1']:.3f} vs "
                f"random {best['mean_random_f1']:.3f} "
                f"(gain {best['mean_model_f1'] - best['mean_random_f1']:+.3f})")

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "bpe_eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out}")
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="BPE composition + boundary-F1 eval on held-out videos")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
