"""
havq/analysis/hierarchy.py
==========================
Bottom-up hierarchical segmentation of ONE video from its LAPA latent actions.

No corpus anywhere: the video's own tokens are centered with the video's own mean,
the tree is built from that video alone, and the per-level vocabulary is fitted on
that video's own segments. LAPA supplies the representation (universal, from
pretraining); everything fitted here is per-video. That is the claim this module
exists to test -- segmenting a lone video with no set of similar videos to learn a
vocabulary from.

Method, per video:
  1. Load the base tokens (<video_id>_emb.npy) and subtract the video's own mean.
     Centering is not optional: the raw deltas carry a constant offset that is ~90%
     of a typical row norm, so uncentered distances measure the offset, not the action.
  2. Merge adjacent segments bottom-up by WARD cost -- repeatedly join the neighbour
     pair whose union stays most internally consistent:
         cost(A,B) = (nA*nB / (nA+nB)) * ||mean_A - mean_B||^2
     Only time-adjacent segments may merge, so every intermediate state is a valid
     segmentation. Ward represents a segment by its member mean, which is exactly the
     pooling the composition check validated (summed fine tokens track the true
     coarse LAPA descriptor: cosine ~0.8 at 1 s, decaying by 8 s -- so pool, rather
     than re-running LAPA at gaps far outside its training distribution).
  3. Each merge deletes exactly ONE boundary, so recording the deletion ORDER
     encodes the whole dendrogram in O(T): after L merges the surviving boundaries
     are the initial ones minus the first L deleted. Every L is one granularity level.
  4. At reported levels: score boundary-F1 against the HD-EPIC narration boundaries
     versus a matched-count random baseline, and fit a per-level vocabulary by
     k-means over that level's segment means (so non-adjacent segments doing the
     same thing can share a token -- merging alone never relates distant segments).

Reads   <features_dir>/lapa_clip<g>_stride<g>/<video_id>{_emb.npy,_meta.json}
Writes  <results_dir>/hierarchy_<video_id>.json   per-level metrics
"""

from __future__ import annotations

import heapq
import json
import logging
import os

import numpy as np
from sklearn.cluster import KMeans

from havq.analysis.bpe_eval import f1, random_f1
from havq.analysis.gt import gt_for_video, load_narrations
from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


def ward_cost(counts, sums, a, b):
    """Ward's increase in within-segment variance from merging segments a and b."""
    diff = sums[a] / counts[a] - sums[b] / counts[b]
    return (counts[a] * counts[b] / (counts[a] + counts[b])) * float(diff @ diff)


def ward_merge_order(X):
    """Contiguity-constrained Ward merging of the rows of X, bottom-up.

    Returns the order in which boundaries are deleted: a list of T-1 token indices,
    where index j names the boundary sitting between token j-1 and token j. After L
    merges the surviving boundaries are {1..T-1} minus the first L entries, so this
    single list encodes every level of the hierarchy.

    Segments live in a doubly linked list so merging is O(1); candidate merges live
    in a heap with lazy invalidation (an entry is stale if either endpoint has been
    absorbed, or they are no longer neighbours).
    """
    n_tokens = len(X)
    counts = np.ones(n_tokens)
    sums = X.astype(np.float64).copy()
    # start[i] = index of the first base token in segment i; also the boundary that
    # disappears when segment i is absorbed into its left neighbour
    start = np.arange(n_tokens)
    next_seg = np.arange(1, n_tokens + 1)
    next_seg[-1] = -1
    prev_seg = np.arange(-1, n_tokens - 1)
    alive = np.ones(n_tokens, dtype=bool)

    heap = []
    for i in range(n_tokens - 1):
        heapq.heappush(heap, (ward_cost(counts, sums, i, i + 1), i, i + 1))

    removal_order = []
    while heap:
        cost, a, b = heapq.heappop(heap)
        # stale entry: an endpoint was absorbed, or they stopped being neighbours
        if not alive[a] or not alive[b] or next_seg[a] != b:
            continue

        removal_order.append(int(start[b]))
        counts[a] += counts[b]
        sums[a] += sums[b]
        alive[b] = False

        next_seg[a] = next_seg[b]
        if next_seg[b] != -1:
            prev_seg[next_seg[b]] = a

        if prev_seg[a] != -1:
            left = prev_seg[a]
            heapq.heappush(heap, (ward_cost(counts, sums, left, a), left, a))
        if next_seg[a] != -1:
            right = next_seg[a]
            heapq.heappush(heap, (ward_cost(counts, sums, a, right), a, right))

    return removal_order


def boundaries_at_level(removal_order, n_tokens, n_merges):
    """Surviving boundaries after n_merges merges, as sorted token indices."""
    gone = set(removal_order[:n_merges])
    return np.array([j for j in range(1, n_tokens) if j not in gone], dtype=np.int64)


def segment_means(X, boundaries):
    """Mean token of each segment cut out by `boundaries`. Returns (n_segments, D)."""
    edges = [0] + list(boundaries) + [len(X)]
    return np.stack([X[edges[i]:edges[i + 1]].mean(axis=0) for i in range(len(edges) - 1)])


def level_vocabulary(means, n_words, seed):
    """k-means over this level's segment means -> one token id per segment.

    This is what lets two NON-ADJACENT segments share a token: Ward only ever
    relates neighbours, so without this step the hierarchy has segments but no
    vocabulary. Returns (labels, reuse) where reuse is the fraction of segments
    whose token is used more than once -- i.e. how much of the video is explained
    by a RECURRING action rather than a one-off.
    """
    n_words = int(min(n_words, len(means)))
    if n_words < 2:
        return np.zeros(len(means), dtype=np.int64), 0.0
    labels = KMeans(n_clusters=n_words, n_init=10, random_state=seed).fit_predict(means)
    _, counts = np.unique(labels, return_counts=True)
    reuse = float(sum(c for c in counts if c > 1) / len(labels))
    return labels.astype(np.int64), reuse


def evaluate_video(X, gt_sec, clips_per_sec, cfg, rng):
    """Build the hierarchy for one video and score every reported level."""
    n_tokens = len(X)
    removal_order = ward_merge_order(X)

    # report levels by SEGMENT COUNT, geometrically spaced: the whole point is the
    # granularity axis, and segment count is the interpretable coordinate on it
    max_segments = min(cfg["eval"]["max_segments"], n_tokens)
    targets = sorted({
        int(round(v))
        for v in np.geomspace(2, max_segments, cfg["eval"]["num_levels"])
    })

    tol = cfg["eval"]["primary_tolerance_sec"]
    levels = []
    for n_segments in targets:
        n_merges = n_tokens - n_segments
        if n_merges < 0 or n_merges > len(removal_order):
            continue
        bounds = boundaries_at_level(removal_order, n_tokens, n_merges)
        pred_sec = bounds / clips_per_sec

        model_f1 = f1(pred_sec, gt_sec, tol)
        rand_f1 = random_f1(len(bounds), n_tokens, gt_sec, clips_per_sec, tol,
                            cfg["eval"]["random_trials"], rng)
        means = segment_means(X, bounds)
        labels, reuse = level_vocabulary(means, cfg["eval"]["words_per_level"], cfg["seed"])

        levels.append({
            "n_segments": int(n_segments),
            "mean_segment_sec": float(n_tokens / clips_per_sec / n_segments),
            "model_f1": float(model_f1),
            "random_f1": float(rand_f1),
            "gain": float(model_f1 - rand_f1),
            "n_words": int(len(np.unique(labels))),
            "token_reuse": float(reuse),
        })
    return levels


def run(cfg):
    features_dir = os.path.join(
        cfg["paths"]["features_dir"],
        feature_subdir(cfg["model"], cfg["pair_gap"], cfg["pair_gap"]),
    )
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    narrations = load_narrations(cfg["paths"]["narrations_pkl"])
    rng = np.random.default_rng(cfg["seed"])

    report = {}
    for video_id in cfg["videos"]:
        with open(os.path.join(features_dir, video_id + "_meta.json")) as f:
            meta = json.load(f)
        X = np.load(os.path.join(features_dir, video_id + "_emb.npy")).astype(np.float64)
        X = X - X.mean(axis=0)  # per-video centering: no corpus needed

        gt_sec, _, _ = gt_for_video(narrations, video_id, meta)
        clips_per_sec = meta["fps"] / meta["n_clip_frame"]

        levels = evaluate_video(X, gt_sec, clips_per_sec, cfg, rng)
        report[video_id] = {"n_tokens": len(X), "n_gt_boundaries": len(gt_sec), "levels": levels}

        logger.info(f"=== {video_id}: {len(X)} tokens, {len(gt_sec)} GT boundaries ===")
        logger.info(f"{'segs':>6}{'avg len':>9}{'F1':>8}{'random':>9}{'gain':>8}{'words':>7}{'reuse':>8}")
        for lv in levels:
            logger.info(
                f"{lv['n_segments']:>6}{lv['mean_segment_sec']:>8.1f}s{lv['model_f1']:>8.3f}"
                f"{lv['random_f1']:>9.3f}{lv['gain']:>+8.3f}{lv['n_words']:>7}{lv['token_reuse']:>8.2f}"
            )
        best = max(levels, key=lambda r: r["gain"])
        logger.info(f"best level: {best['n_segments']} segments, gain {best['gain']:+.3f}")

    out = os.path.join(results_dir, "hierarchy_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out}")
    return report
