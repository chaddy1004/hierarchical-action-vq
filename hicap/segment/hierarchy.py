"""
hicap/segment/hierarchy.py
==========================
Contiguity-constrained agglomerative segmentation of one video from a per-frame
(or per-clip) feature sequence. Bottom-up: start with every frame its own
segment, repeatedly merge the adjacent pair with the lowest linkage cost, and
record the order boundaries disappear.

Ported from the verified Ward implementation in
archive/v2-vjepa-lapa/havq/analysis/hierarchy.py (exact-recovery tested on
synthetic block signals, incl. unequal blocks). The only additions here are a
selectable linkage (`ward` | `average`) and optional L2-normalization so cosine
geometry is available for normalized video features.

Every merge deletes exactly ONE boundary, so `ward_merge_order` returns the
deletion order and a single list encodes the whole dendrogram: after L merges the
surviving boundaries are {1..T-1} minus the first L deleted. Every L is a level.
"""

from __future__ import annotations

import heapq

import numpy as np


def ward_cost(counts, sums, a, b):
    """Ward's increase in within-segment variance from merging segments a and b."""
    diff = sums[a] / counts[a] - sums[b] / counts[b]
    return (counts[a] * counts[b] / (counts[a] + counts[b])) * float(diff @ diff)


def average_cost(counts, sums, a, b):
    """Squared distance between the two segment means (group-average linkage)."""
    diff = sums[a] / counts[a] - sums[b] / counts[b]
    return float(diff @ diff)


LINKAGES = {"ward": ward_cost, "average": average_cost}


def merge_order(X, linkage="ward", l2_normalize=False):
    """Contiguity-constrained agglomerative merging of the rows of X, bottom-up.

    X:            (T, D) feature sequence (one row per frame/clip).
    linkage:      'ward' (variance) or 'average' (mean-to-mean distance).
    l2_normalize: unit-normalize rows first, so distances are cosine geometry.

    Returns the order in which boundaries are deleted: a list of T-1 token
    indices, where index j names the boundary between token j-1 and token j.
    After L merges the surviving boundaries are {1..T-1} minus the first L entries.

    Segments live in a doubly linked list so a merge is O(1); candidate merges
    live in a heap with lazy invalidation (an entry is stale if either endpoint
    was absorbed, or they are no longer neighbours).
    """
    cost_fn = LINKAGES[linkage]
    X = np.asarray(X, dtype=np.float64)
    if l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = X / norms

    n_tokens = len(X)
    counts = np.ones(n_tokens)
    sums = X.copy()
    # start[i] = first base token of segment i; also the boundary that disappears
    # when segment i is absorbed into its left neighbour
    start = np.arange(n_tokens)
    next_seg = np.arange(1, n_tokens + 1)
    next_seg[-1] = -1
    prev_seg = np.arange(-1, n_tokens - 1)
    alive = np.ones(n_tokens, dtype=bool)

    heap = []
    for i in range(n_tokens - 1):
        heapq.heappush(heap, (cost_fn(counts, sums, i, i + 1), i, i + 1))

    removal_order = []
    while heap:
        _, a, b = heapq.heappop(heap)
        if not alive[a] or not alive[b] or next_seg[a] != b:
            continue  # stale

        removal_order.append(int(start[b]))
        counts[a] += counts[b]
        sums[a] += sums[b]
        alive[b] = False

        next_seg[a] = next_seg[b]
        if next_seg[b] != -1:
            prev_seg[next_seg[b]] = a

        if prev_seg[a] != -1:
            left = prev_seg[a]
            heapq.heappush(heap, (cost_fn(counts, sums, left, a), left, a))
        if next_seg[a] != -1:
            right = next_seg[a]
            heapq.heappush(heap, (cost_fn(counts, sums, a, right), a, right))

    return removal_order


def boundaries_at_level(removal_order, n_tokens, n_merges):
    """Surviving boundaries after n_merges merges, as sorted token indices."""
    gone = set(removal_order[:n_merges])
    return np.array([j for j in range(1, n_tokens) if j not in gone], dtype=np.int64)


def segmentation_at_k(removal_order, n_tokens, n_segments):
    """Frame-wise segment id when the video is cut into exactly n_segments pieces.

    Returns an int array (n_tokens,) labeling each frame 0..n_segments-1 in
    temporal order -- the flat partition at that granularity level.
    """
    n_merges = n_tokens - n_segments
    bounds = boundaries_at_level(removal_order, n_tokens, n_merges)
    labels = np.zeros(n_tokens, dtype=np.int64)
    edges = [0] + list(bounds) + [n_tokens]
    for seg_id in range(len(edges) - 1):
        labels[edges[seg_id]:edges[seg_id + 1]] = seg_id
    return labels
