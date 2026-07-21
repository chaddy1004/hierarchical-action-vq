"""
hicap/eval/baselines.py
=======================
Unsupervised segmentation baselines, each mapping a feature sequence to one
cluster id per frame given a target cluster count C. Evaluated with the same
standard metrics as our method (hicap/eval/tas_metrics.py), on the same features,
so the comparison is apples-to-apples.

  uniform_labels  -- C equal-length contiguous blocks. No features at all; the
                     trivial floor for the "known K" setting.
  kmeans_labels   -- KMeans(C) on frame features. Allows a class to recur on
                     non-adjacent segments, but has no temporal structure.
  twfinch_labels  -- TW-FINCH (Sarfraz et al., CVPR 2021): FINCH first-neighbour
                     clustering on a temporally-weighted distance
                     d(i,j) = (1 - cos(f_i, f_j)) * |i - j| / T, agglomeratively
                     reduced to exactly C clusters. The standard strong
                     unsupervised-TAS baseline (temporal + feature).

Clustering runs on frames subsampled by `stride` (the O(T^2) TW distance is
otherwise ~T=12k per 50 Salads video); labels are block-upsampled back to full
frame resolution so all metrics are computed at native resolution.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # noqa: F401  (kept: metrics import chain)
from sklearn.cluster import AgglomerativeClustering, KMeans


def subsample(features, stride):
    idx = np.arange(0, len(features), stride)
    return features[idx], idx


def upsample_labels(sub_labels, sub_idx, n_frames):
    """Assign each full-res frame the label of the nearest sampled frame."""
    full = np.zeros(n_frames, dtype=np.int64)
    for k, start in enumerate(sub_idx):
        end = sub_idx[k + 1] if k + 1 < len(sub_idx) else n_frames
        full[start:end] = sub_labels[k]
    return full


def uniform_labels(n_frames, n_clusters):
    edges = np.linspace(0, n_frames, n_clusters + 1).astype(int)
    labels = np.zeros(n_frames, dtype=np.int64)
    for c in range(n_clusters):
        labels[edges[c]:edges[c + 1]] = c
    return labels


def kmeans_labels(features, n_clusters, stride=10, seed=0):
    sub, idx = subsample(features, stride)
    k = min(n_clusters, len(sub))
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(sub)
    return upsample_labels(lab, idx, len(features))


def tw_distance(unit_means, time_means, n):
    """Temporally-weighted cosine distance between cluster means (TW-FINCH):
    d(i,j) = (1 - cos(f_i, f_j)) * |t_i - t_j| / n. Diagonal set to +inf."""
    feat = 1.0 - unit_means @ unit_means.T
    temporal = np.abs(time_means[:, None] - time_means[None, :]) / float(n)
    d = feat * temporal
    np.fill_diagonal(d, np.inf)
    return d


def twfinch_labels(features, n_clusters, stride=10):
    """TW-FINCH reduced to exactly n_clusters, per full-res frame.

    Faithful recursion: repeatedly build the FINCH first-neighbour partition under
    the temporally-weighted distance (on cluster means), coarsening until the next
    step would drop below n_clusters; then merge to EXACTLY n_clusters with
    agglomeration on the SAME temporally-weighted distance (precomputed), so the
    final clusters stay temporally coherent instead of scattering.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    sub, idx = subsample(features, stride)
    n = len(sub)
    if n <= n_clusters:
        return upsample_labels(np.arange(n), idx, len(features))

    unit = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    time = np.arange(n, dtype=float)
    labels = np.arange(n)  # current partition of the n subsampled frames

    def means_of(lab):
        ids = np.unique(lab)
        um = np.stack([unit[lab == c].mean(0) for c in ids])
        um /= np.linalg.norm(um, axis=1, keepdims=True) + 1e-8
        tm = np.array([time[lab == c].mean() for c in ids])
        return ids, um, tm

    # coarsen via FINCH first-neighbour partitions until near the target
    while True:
        ids, um, tm = means_of(labels)
        m = len(ids)
        if m <= n_clusters:
            break
        d = tw_distance(um, tm, n)
        nn = d.argmin(axis=1)
        adj = csr_matrix((np.ones(m), (np.arange(m), nn)), shape=(m, m))
        adj = adj + adj.T
        n_comp, comp = connected_components(adj, directed=False)
        if n_comp < n_clusters:
            # this step overshoots: merge current means to EXACTLY n_clusters
            d_full = d.copy()
            np.fill_diagonal(d_full, 0.0)
            merged = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            ).fit_predict(d_full)
            labels = merged[np.searchsorted(ids, labels)]
            break
        if n_comp == m:
            break  # no further merging possible
        labels = comp[np.searchsorted(ids, labels)]

    return upsample_labels(labels, idx, len(features))
