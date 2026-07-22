"""
hicap/eval/gate.py
==================
Experiment 1 gate (hicap/PLAN.md): on 50 Salads, does the agglomerative hierarchy
contain a granularity level whose boundaries beat BOTH uniform segmentation and a
matched-count random baseline, against the real annotations -- without ever
consuming the ground-truth segment count?

Per video: build one hierarchy from the frame features, then read it at a sweep of
segment counts (levels). At each level score boundary-F1 vs the GT boundaries and
against the two matched-count baselines. Aggregate across videos and, for every
GT granularity present, report the F1-vs-level curve and the level with the best
mean gain over uniform.

Decision gate: some level beats uniform (and random) on >= the mid-level GT ->
the spine holds. No level beats uniform -> the grouping is the problem; fix
boundary detection / linkage before adding datasets or captioning.

Writes <results_dir>/gate_report.json.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from hicap.data import tas
from hicap.eval.boundary import boundary_f1, gt_boundaries, random_f1, uniform_f1
from hicap.segment.hierarchy import boundaries_at_level, merge_order

logger = logging.getLogger(__name__)


def level_targets(n_frames, min_seg, max_segments, num_levels):
    """Segment counts to report, geometric between 2 and max_segments."""
    top = min(max_segments, n_frames // max(min_seg, 1))
    if top < 2:
        return []
    return sorted({int(round(v)) for v in np.geomspace(2, top, num_levels)})


def evaluate_video(features, labels_by_gran, cfg, fps, rng):
    """Build the hierarchy once; score every level against every GT granularity."""
    n_frames = len(features)
    order = merge_order(
        features,
        linkage=cfg["linkage"],
        l2_normalize=cfg["l2_normalize"],
    )
    tol = int(round(cfg["tol_sec"] * fps))
    targets = level_targets(n_frames, cfg["min_segment_frames"], cfg["max_segments"], cfg["num_levels"])

    gts = {g: gt_boundaries(lab) for g, lab in labels_by_gran.items()}

    per_gran = {g: [] for g in labels_by_gran}
    for n_segments in targets:
        n_merges = n_frames - n_segments
        if n_merges < 0:
            continue
        pred = boundaries_at_level(order, n_frames, n_merges)
        for g, gt in gts.items():
            per_gran[g].append({
                "n_segments": int(n_segments),
                "model_f1": boundary_f1(pred, gt, tol),
                "uniform_f1": uniform_f1(len(pred), n_frames, gt, tol),
                "random_f1": random_f1(len(pred), n_frames, gt, tol, cfg["random_trials"], rng),
                "n_gt": int(len(gt)),
            })
    return per_gran


def aggregate(per_video, granularity):
    """Mean curves across videos for one granularity, keyed by segment count."""
    by_level = {}
    for vid_levels in per_video:
        for lv in vid_levels[granularity]:
            by_level.setdefault(lv["n_segments"], []).append(lv)
    curve = []
    for n_segments in sorted(by_level):
        rows = by_level[n_segments]
        m = float(np.mean([r["model_f1"] for r in rows]))
        u = float(np.mean([r["uniform_f1"] for r in rows]))
        r = float(np.mean([r["random_f1"] for r in rows]))
        curve.append({
            "n_segments": n_segments,
            "model_f1": m, "uniform_f1": u, "random_f1": r,
            "gain_vs_uniform": m - u, "gain_vs_random": m - r,
            "n_videos": len(rows),
        })
    return curve


def run(cfg):
    root = cfg["paths"]["data_root"]
    dataset = cfg["dataset"]
    fps = tas.fps(dataset)
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    videos = tas.list_videos(root, dataset)
    if cfg.get("max_videos"):
        videos = videos[: cfg["max_videos"]]
    grans = list(cfg.get("granularities") or ["groundTruth"])
    finest_gran = "groundTruth" if "groundTruth" in grans else grans[0]
    report_grans = grans + (["verb"] if cfg.get("derive_verb_level") else [])
    logger.info(f"Gate on {dataset}: {len(videos)} videos ({fps} fps) | granularities {grans} | "
                f"linkage={cfg['linkage']} l2={cfg['l2_normalize']} tol={cfg['tol_sec']}s")

    per_video = []
    for i, vid in enumerate(videos):
        features = tas.load_features(root, dataset, vid)
        labels_by_gran = {g: tas.load_labels(root, dataset, vid, g) for g in grans}
        if cfg.get("derive_verb_level"):
            labels_by_gran["verb"] = tas.derive_verb_level(labels_by_gran[finest_gran])
        per_video.append(evaluate_video(features, labels_by_gran, cfg, fps, rng))
        logger.info(f"  [{i + 1}/{len(videos)}] {vid}: {len(features)} frames")

    report = {"config": cfg, "granularities": {}}
    for g in report_grans:
        curve = aggregate(per_video, g)
        best = max(curve, key=lambda r: r["gain_vs_uniform"]) if curve else None
        report["granularities"][g] = {"curve": curve, "best_vs_uniform": best}
        if best:
            logger.info(f"[{g}] best level {best['n_segments']} segs: "
                        f"model {best['model_f1']:.3f} | uniform {best['uniform_f1']:.3f} "
                        f"(gain {best['gain_vs_uniform']:+.3f}) | random {best['random_f1']:.3f}")

    out = os.path.join(results_dir, f"gate_report_{dataset}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out}")

    verdict_lines = []
    for g, gd in report["granularities"].items():
        b = gd["best_vs_uniform"]
        if b and b["gain_vs_uniform"] > 0 and b["gain_vs_random"] > 0:
            verdict_lines.append(f"[{g}] PASS: beats uniform (+{b['gain_vs_uniform']:.3f}) and random at {b['n_segments']} segs")
        else:
            verdict_lines.append(f"[{g}] FAIL: no level beats uniform (best gain {b['gain_vs_uniform']:+.3f})" if b else f"[{g}] no levels")
    for line in verdict_lines:
        logger.info(line)
    return report
