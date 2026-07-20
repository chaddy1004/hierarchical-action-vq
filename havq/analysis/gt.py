"""
havq/analysis/gt.py
===================
HD-EPIC ground truth mapped into clip-index space, for a given feature cache.
Ported from archive/v1-derisk/havq/gt.py; the only substantive change is the
meta key -- v1 wrote `delta_min`, the current preprocessing writes
`n_clip_frame` (havq/data/preprocessing/preprocessing.py) -- plus reading the
feature directory via havq.utils.paths.feature_subdir.

Clip j covers frames [j*n_clip_frame, (j+1)*n_clip_frame) at the video's fps,
so clip center time = (j + 0.5) * n_clip_frame / fps. Per video this caches
<feature_subdir>/<video_id>_gt.npz with:

    boundaries_sec : float array, sorted unique narration start/end times (sec)
    boundaries     : int array, the same snapped to clip indices (floor), in
                     [1, n_clips-1] -- convenience for clip-space overlays
    labels         : int array (n_clips,), main verb class of the narration
                     covering the clip center; -1 = background (no narration).

Label = first entry of `verb_classes` (the narration's main verb). Where
narrations overlap, the later-starting narration wins.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle

import numpy as np

from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


def load_narrations(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def gt_for_video(narrations, video_id: str, meta: dict):
    """Build GT boundaries and per-clip verb labels for one video."""
    fps = meta["fps"]
    n_clips = meta["n_clips"]
    n_clip_frame = meta["n_clip_frame"]
    clips_per_sec = fps / n_clip_frame
    duration_sec = n_clips / clips_per_sec  # end of the last full clip

    rows = narrations[narrations["video_id"] == video_id]
    if len(rows) == 0:
        raise ValueError(f"No narrations for video_id={video_id!r}")
    rows = rows.sort_values("start_timestamp")

    boundaries_sec = set()
    labels = np.full(n_clips, -1, dtype=np.int64)
    clip_centers_sec = (np.arange(n_clips) + 0.5) / clips_per_sec

    for _, row in rows.iterrows():
        for ts in (float(row["start_timestamp"]), float(row["end_timestamp"])):
            if 0.0 < ts < duration_sec:
                boundaries_sec.add(ts)
        vcs = row["verb_classes"]
        if vcs is None or len(vcs) == 0:
            continue  # unlabeled narration: contributes boundaries only
        covered = (clip_centers_sec >= row["start_timestamp"]) & (
            clip_centers_sec < row["end_timestamp"]
        )
        labels[covered] = int(vcs[0])

    boundaries_sec = np.array(sorted(boundaries_sec), dtype=np.float64)
    boundaries = np.unique((boundaries_sec * clips_per_sec).astype(np.int64))
    boundaries = boundaries[(boundaries >= 1) & (boundaries <= n_clips - 1)]

    return boundaries_sec, boundaries, labels


def build_all(cfg: dict, overwrite: bool = False) -> None:
    features_root = cfg["paths"]["features_root"]
    subdir = feature_subdir(cfg["model"], cfg["scale"]["n_clip_frame"], cfg["scale"]["stride"])
    features_dir = os.path.join(features_root, subdir)
    narrations = load_narrations(cfg["paths"]["narrations_pkl"])

    meta_files = sorted(f for f in os.listdir(features_dir) if f.endswith("_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No feature meta files in {features_dir}")

    n_done = n_skip = 0
    for meta_file in meta_files:
        video_id = meta_file[: -len("_meta.json")]
        out_npz = os.path.join(features_dir, video_id + "_gt.npz")
        if os.path.exists(out_npz) and not overwrite:
            n_skip += 1
            continue
        if int((narrations["video_id"] == video_id).sum()) == 0:
            logger.warning(f"{video_id}: no narrations found -- skipping GT")
            continue

        with open(os.path.join(features_dir, meta_file)) as f:
            meta = json.load(f)
        boundaries_sec, boundaries, labels = gt_for_video(narrations, video_id, meta)
        np.savez(out_npz, boundaries_sec=boundaries_sec, boundaries=boundaries, labels=labels)

        n_bg = int((labels == -1).sum())
        logger.info(
            f"{video_id}: {meta['n_clips']} clips, {len(boundaries_sec)} GT boundaries, "
            f"background {n_bg}/{meta['n_clips']} ({100.0 * n_bg / meta['n_clips']:.0f}%)"
        )
        n_done += 1

    logger.info(f"Built GT for {n_done} videos ({n_skip} already cached) in {features_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build per-clip verb labels + boundaries for a feature cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="rebuild cached GT files")
    args = parser.parse_args()
    build_all(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
