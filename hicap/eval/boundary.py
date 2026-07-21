"""
hicap/eval/boundary.py
======================
Boundary-detection F1 with matched-budget baselines, for the count-free gate.

A predicted boundary counts as correct if it falls within `tol` frames of a
ground-truth boundary; each GT boundary matches at most one prediction (greedy by
distance). F1 is the harmonic mean of the resulting precision and recall.

The point of this module is the two baselines, both placed with the SAME number
of boundaries the model proposed at that level -- so F1 gains cannot come from
simply cutting more often:
  - uniform: boundaries at equal spacing (no model at all). The hard baseline;
    on near-periodic annotations it is close to optimal, and it beat an
    agglomerative hierarchy on 2/3 HD-EPIC videos in the earlier PoC.
  - random:  boundaries at uniformly random positions, averaged over trials. The
    floor.

Boundary/F1 machinery ported from archive/v2-vjepa-lapa/havq/analysis/bpe_eval.py.
"""

from __future__ import annotations

import numpy as np


def gt_boundaries(labels):
    """Frame indices where a per-frame label sequence changes value."""
    labels = np.asarray(labels)
    return np.where(labels[1:] != labels[:-1])[0] + 1


def match_count(pred, gt, tol):
    """Greedy #matches: each pred and each gt used at most once, |pred-gt| <= tol."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = abs(int(p) - int(g))
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp


def boundary_f1(pred, gt, tol):
    tp = match_count(pred, gt, tol)
    prec = tp / len(pred) if len(pred) else 0.0
    rec = tp / len(gt) if len(gt) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def uniform_boundaries(n_pred, n_frames):
    """n_pred boundaries at equal spacing over a video of n_frames frames."""
    if n_pred <= 0:
        return np.array([], dtype=np.int64)
    edges = np.linspace(0, n_frames, n_pred + 2)[1:-1]
    return np.clip(np.round(edges).astype(np.int64), 1, n_frames - 1)


def uniform_f1(n_pred, n_frames, gt, tol):
    return boundary_f1(uniform_boundaries(n_pred, n_frames), gt, tol)


def random_f1(n_pred, n_frames, gt, tol, trials, rng):
    """Mean F1 of n_pred boundaries placed at uniformly random distinct positions."""
    if n_pred <= 0 or n_frames <= 1:
        return 0.0
    k = min(n_pred, n_frames - 1)
    scores = [
        boundary_f1(np.sort(rng.choice(np.arange(1, n_frames), size=k, replace=False)), gt, tol)
        for _ in range(trials)
    ]
    return float(np.mean(scores))
