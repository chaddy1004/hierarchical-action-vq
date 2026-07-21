"""
hicap/eval/tas_metrics.py
=========================
Standard temporal-action-segmentation metrics, so hicap results are comparable to
the TAS literature (MS-TCN, CTE, TW-FINCH, ...). All operate on per-frame label
sequences (predictions and ground truth of equal length).

  mof_hungarian  -- frame accuracy after optimally matching predicted CLUSTER ids
                    to GT class ids (Hungarian). The headline unsupervised-TAS
                    number: predictions are cluster ids with no inherent class
                    meaning, so they are matched to GT once, globally per video.
  edit_score     -- normalized segmental Levenshtein (order correctness).
  f_score        -- segmental F1 at an IoU overlap threshold (F1@{10,25,50}).

Segmental metrics (edit, F1) are computed on the run-length segment sequences.
Implementations follow the standard MS-TCN eval (github.com/yabufarha/ms-tcn),
reimplemented here so the package stands alone.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def to_int_labels(frame_labels):
    """Map an array of (possibly string) per-frame labels to 0..C-1 ints."""
    classes = {c: i for i, c in enumerate(sorted(set(map(str, frame_labels))))}
    return np.array([classes[str(x)] for x in frame_labels]), classes


def segments(frame_labels):
    """Run-length encode: list of (label, start, end_exclusive)."""
    labels = np.asarray(frame_labels)
    if len(labels) == 0:
        return []
    out = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            out.append((labels[start], start, i))
            start = i
    out.append((labels[start], start, len(labels)))
    return out


def mof_hungarian(pred_frame, gt_frame):
    """Frame accuracy after Hungarian-matching predicted clusters to GT classes.

    pred_frame: int cluster id per frame. gt_frame: label per frame (any hashable).
    Returns (mof, matched_pred_frame) where matched_pred_frame carries the GT class
    each frame's cluster was matched to (for downstream segmental metrics).
    """
    gt_int, _ = to_int_labels(gt_frame)
    pred = np.asarray(pred_frame)
    n_pred = int(pred.max()) + 1 if len(pred) else 0
    n_gt = int(gt_int.max()) + 1 if len(gt_int) else 0

    # contingency[i, j] = frames with predicted cluster i and GT class j
    contingency = np.zeros((n_pred, n_gt), dtype=np.int64)
    for p, g in zip(pred, gt_int):
        contingency[p, g] += 1

    # maximize total matched frames -> minimize the negative
    row, col = linear_sum_assignment(-contingency)
    cluster_to_class = {int(r): int(c) for r, c in zip(row, col)}
    matched = np.array([cluster_to_class.get(int(p), -1) for p in pred])
    mof = float((matched == gt_int).mean()) if len(pred) else 0.0
    return mof, matched, gt_int


def edit_score(pred_frame, gt_frame, norm=True):
    """Normalized segmental edit (Levenshtein) distance -> similarity in [0,100]."""
    p = [s[0] for s in segments(pred_frame)]
    g = [s[0] for s in segments(gt_frame)]
    m, n = len(p), len(g)
    if m == 0 and n == 0:
        return 100.0
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    dist = d[m, n]
    if norm:
        return (1.0 - dist / max(m, n)) * 100.0
    return dist


def f_score(pred_frame, gt_frame, overlap):
    """Segmental F1 at IoU threshold `overlap` (e.g. 0.1, 0.25, 0.5), in [0,100].

    Each predicted segment is a true positive if it overlaps a same-label GT
    segment with IoU >= overlap and that GT segment is not already claimed.
    """
    p_segs = segments(pred_frame)
    g_segs = segments(gt_frame)
    tp = 0
    used = np.zeros(len(g_segs), dtype=bool)
    for pl, ps, pe in p_segs:
        best_iou, best_j = 0.0, -1
        for j, (gl, gs, ge) in enumerate(g_segs):
            if gl != pl or used[j]:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= overlap:
            tp += 1
            used[best_j] = True
    n_pred, n_gt = len(p_segs), len(g_segs)
    fp = n_pred - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) * 100.0 if (prec + rec) else 0.0


def all_metrics(pred_frame, gt_frame, overlaps=(0.1, 0.25, 0.5)):
    """MoF + edit + F1@overlaps for one video. pred_frame = cluster ids per frame."""
    mof, matched, _ = mof_hungarian(pred_frame, gt_frame)
    # segmental metrics use the Hungarian-matched labels vs GT ints
    gt_int, _ = to_int_labels(gt_frame)
    out = {"mof": mof * 100.0, "edit": edit_score(matched, gt_int)}
    for ov in overlaps:
        out[f"f1@{int(ov * 100)}"] = f_score(matched, gt_int, ov)
    return out
