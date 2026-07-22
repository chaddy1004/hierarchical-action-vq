"""
hicap/eval/compare.py
=====================
Standard "known-K" unsupervised-TAS comparison on 50 Salads: put every method on
the same footing -- each produces exactly C clusters (C = #unique GT actions in
the video) -- and score with the same metrics (MoF/edit/F1@{10,25,50}).

Methods:
  uniform        -- C equal contiguous blocks (floor).
  kmeans         -- KMeans(C) on frames (feature-only).
  twfinch        -- TW-FINCH (the standard strong unsupervised-TAS baseline).
  ours_contig    -- our hierarchy cut to C contiguous segments (pure count-free
                    readout; no recurrence -- one label per contiguous piece).
  ours_cluster   -- our hierarchy over-segmented then grouped to C (recurrence-
                    aware readout, comparable to the frame-clustering baselines).

This is the "known K" table for literature comparison. It does NOT use the
count-free curve (that lives in eval/gate.py); here every method is handed C, so
the comparison isolates segmentation quality from count selection.

Writes <results_dir>/compare_report.json.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from hicap.data import tas
from hicap.eval import baselines
from hicap.eval.tas_metrics import all_metrics
from hicap.segment.hierarchy import cluster_at_k, merge_order, segmentation_at_k

logger = logging.getLogger(__name__)

METHODS = ["uniform", "kmeans", "twfinch", "ours_contig", "ours_cluster"]


def methods_for_video(features, n_clusters, cfg):
    """Return {method: per-frame cluster-id array} for one video at cluster count C."""
    T = len(features)
    order = merge_order(features, linkage=cfg["linkage"], l2_normalize=cfg["l2_normalize"])
    overseg = int(cfg["oversegment_factor"] * n_clusters)
    return {
        "uniform": baselines.uniform_labels(T, n_clusters),
        "kmeans": baselines.kmeans_labels(features, n_clusters, cfg["cluster_stride"], cfg["seed"]),
        "twfinch": baselines.twfinch_labels(features, n_clusters, cfg["cluster_stride"]),
        "ours_contig": segmentation_at_k(order, T, n_clusters),
        "ours_cluster": cluster_at_k(features, order, overseg, n_clusters, cfg["seed"]),
    }


def run(cfg):
    root = cfg["paths"]["data_root"]
    dataset = cfg["dataset"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    videos = tas.list_videos(root, dataset)
    if cfg.get("max_videos"):
        videos = videos[: cfg["max_videos"]]
    logger.info(f"Known-K comparison on {dataset}: {len(videos)} videos | methods {METHODS}")

    per_method = {m: [] for m in METHODS}
    for i, vid in enumerate(videos):
        features = tas.load_features(root, dataset, vid)
        gt = tas.load_labels(root, dataset, vid, cfg["granularity"])
        n_clusters = len(set(gt.tolist()))
        preds = methods_for_video(features, n_clusters, cfg)
        for m in METHODS:
            per_method[m].append(all_metrics(preds[m], gt))
        logger.info(f"  [{i + 1}/{len(videos)}] {vid}: C={n_clusters}, {len(features)} frames")

    report = {"config": cfg, "n_videos": len(videos), "methods": {}}
    metric_keys = ["mof", "edit", "f1@10", "f1@25", "f1@50"]
    for m in METHODS:
        report["methods"][m] = {k: float(np.mean([r[k] for r in per_method[m]])) for k in metric_keys}

    out = os.path.join(results_dir, f"compare_report_{dataset}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 66)
    logger.info(f"{'method':<14}" + "".join(f"{k:>10}" for k in metric_keys))
    for m in METHODS:
        r = report["methods"][m]
        logger.info(f"{m:<14}" + "".join(f"{r[k]:>10.1f}" for k in metric_keys))
    logger.info(f"Wrote {out}")
    return report
