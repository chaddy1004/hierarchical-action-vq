"""
hicap/data/salads.py
====================
50 Salads reader for the count-free boundary gate.

Expects the canonical MS-TCN layout (the Zenodo mirror, record 3625992), which
every unsupervised-TAS baseline uses -- so comparisons here are apples-to-apples:

    <root>/50salads/
        features/<video_id>.npy      (D, T) I3D features, dim-first
        groundTruth/<video_id>.txt   T lines, one action label (string) per frame
        mapping.txt                  "<id> <classname>" per line
        splits/{train,test}.split<k>.bundle   video-id lists (one <video_id>.txt per line)

Features are returned transposed to (T, D) so rows are frames -- the convention
the hierarchy builder expects. Per-frame label strings are returned as-is; the
gate turns them into boundaries with hicap.eval.boundary.gt_boundaries.

50 Salads is 30 fps; boundary tolerances in the gate config are given in seconds
and converted with FPS here.
"""

from __future__ import annotations

import os

import numpy as np

FPS = 30.0


def dataset_dir(root):
    """Accept either the parent of 50salads/ or the 50salads/ dir itself."""
    if os.path.isdir(os.path.join(root, "50salads")):
        return os.path.join(root, "50salads")
    return root


def list_videos(root):
    d = dataset_dir(root)
    feat_dir = os.path.join(d, "features")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(feat_dir) if f.endswith(".npy"))


def load_features(root, video_id):
    """(T, D) float32 feature sequence for one video (rows = frames)."""
    d = dataset_dir(root)
    arr = np.load(os.path.join(d, "features", video_id + ".npy"))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (D, T) -> (T, D)
    return arr.astype(np.float32)


def load_labels(root, video_id, granularity="groundTruth"):
    """Per-frame action-label strings for one video, at the given granularity dir."""
    d = dataset_dir(root)
    with open(os.path.join(d, granularity, video_id + ".txt")) as f:
        return np.array([ln.strip() for ln in f if ln.strip() != ""])


def available_granularities(root):
    """Label directories present besides features/splits -- each is one GT level."""
    d = dataset_dir(root)
    skip = {"features", "splits"}
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isdir(p) and name not in skip and any(f.endswith(".txt") for f in os.listdir(p)):
            out.append(name)
    return out
