"""
hicap/data/tas.py
=================
Dataset-agnostic reader for the standard temporal-action-segmentation benchmarks
in the MS-TCN release (Zenodo 3625992): 50 Salads, Breakfast, GTEA. All three
share one on-disk layout, so one reader serves all three:

    <root>/<dataset>/
        features/<video_id>.npy      (D, T) I3D features, dim-first
        groundTruth/<video_id>.txt   T lines, one action label (string) per frame
        mapping.txt                  "<id> <classname>" per line
        splits/...                   video-id bundles (unused here)

`root` is the parent of the per-dataset dirs (e.g. .../data). `dataset` is one of
DATASETS. Features are returned (T, D) so rows are frames. The only per-dataset
constant that matters downstream is the frame rate, used to convert a boundary
tolerance in seconds to frames.

Supersedes the earlier salads-only reader; same functions, plus a `dataset` arg.
"""

from __future__ import annotations

import os

import numpy as np

# frame rate per dataset. 50 Salads is 30 fps; Breakfast and GTEA I3D features are
# at 15 fps. Used only to convert tolerance-in-seconds to tolerance-in-frames.
DATASETS = {
    "50salads": {"fps": 30.0},
    "breakfast": {"fps": 15.0},
    "gtea": {"fps": 15.0},
}

# I3D feature dimensionality. MS-TCN stores features dim-first as (D, T); we key on
# this to transpose to (T, D). A shape heuristic (T > D) is WRONG for short videos:
# Breakfast/GTEA clips have T < 2048, so (2048, T) would be left untransposed.
FEATURE_DIM = 2048


def fps(dataset):
    return DATASETS[dataset]["fps"]


def dataset_dir(root, dataset):
    return os.path.join(root, dataset)


def list_videos(root, dataset):
    feat_dir = os.path.join(dataset_dir(root, dataset), "features")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(feat_dir) if f.endswith(".npy"))


def load_features(root, dataset, video_id):
    """(T, D) float32 feature sequence for one video (rows = frames).

    MS-TCN stores features dim-first (D, T); we transpose to (T, D). The dim axis
    is FEATURE_DIM (2048), used explicitly rather than a T>D heuristic that breaks
    on short (Breakfast/GTEA) clips.
    """
    arr = np.load(os.path.join(dataset_dir(root, dataset), "features", video_id + ".npy"))
    if arr.shape[0] == FEATURE_DIM and arr.shape[1] != FEATURE_DIM:
        arr = arr.T  # (D, T) -> (T, D)
    return arr.astype(np.float32)


def load_labels(root, dataset, video_id, granularity="groundTruth"):
    """Per-frame action-label strings for one video."""
    path = os.path.join(dataset_dir(root, dataset), granularity, video_id + ".txt")
    with open(path) as f:
        return np.array([ln.strip() for ln in f if ln.strip() != ""])


def load_labels_strided(root, dataset, video_id, stride, window, n_clips, granularity="groundTruth"):
    """Per-frame GT downsampled to clip resolution, for clip-level features
    (e.g. V-JEPA). Clip j pools frames [j*stride, j*stride+window); we label it by
    the class at the window centre. Returns an array of length n_clips, aligned
    row-for-row with a clip-resolution feature sequence so the same gate/compare
    machinery applies. `granularity` still selects the GT level.
    """
    labels = load_labels(root, dataset, video_id, granularity)
    centres = np.clip(np.arange(n_clips) * stride + window // 2, 0, len(labels) - 1)
    return labels[centres]


def derive_verb_level(labels):
    """Coarsen per-frame labels to their verb prefix: pour_milk -> pour.

    A reproducible coarser granularity from the labels alone (no extra
    annotations). All three datasets use verb_object class names (50 Salads
    cut_tomato, Breakfast pour_cereals, GTEA take_cup), so the first underscore
    token is the action verb. Background tokens (SIL, action_start) coarsen to
    themselves, which is fine.
    """
    return np.array([str(x).split("_")[0] for x in labels])
