"""
havq/gt.py
==========
HD-EPIC ground truth mapped into clip-index space.

Clip-index space is defined by the cached V-JEPA features (havq/features.py):
clip j covers frames [j*delta_min, (j+1)*delta_min), i.e. the time interval
[j*delta_min/fps, (j+1)*delta_min/fps). fps and n_clips are read from the
feature meta json, so features must be extracted before GT can be built.

Per video this produces (cached to <features_dir>/<video_id>_gt.npz):

    boundaries_sec : float array, sorted unique narration start/end timestamps
                     in seconds — the CANONICAL eval target. GT stays at
                     annotation resolution; only predictions are quantized
                     (standard TAS convention: tolerance absorbs the method's
                     own grid, GT is never degraded to it).
    boundaries     : int array, the same boundaries snapped to clip indices
                     (floor), in [1, n_clips-1]. Convenience only, for quick
                     clip-space overlays/diagnostics.
    labels         : int array (n_clips,), main verb class of the narration
                     covering the clip center; -1 = background (no narration).
                     Per-clip by design (answers "which action owns each clip").

Label = first entry of `verb_classes` (the narration's main verb). Where
narrations overlap, the later-starting narration wins.

CLI prints a per-video summary (segment counts, background fraction, top
verbs) for spot-checking against the raw narrations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from havq import load_config

logger = logging.getLogger(__name__)


def load_narrations(pkl_path: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_verb_names(csv_path: str) -> dict:
    """Map verb class id -> canonical verb name ('key' column)."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["id"].astype(int), df["key"]))


def load_meta(features_dir: str, video_id: str) -> dict:
    meta_path = os.path.join(features_dir, video_id + "_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — run havq/features.py first (GT lives in "
            f"clip-index space, which is defined by the cached features)."
        )
    with open(meta_path, "r") as f:
        return json.load(f)


def gt_for_video(
    narrations: pd.DataFrame, video_id: str, meta: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build GT boundaries and per-clip labels for one video.

    Returns
    -------
    boundaries_sec : np.ndarray float64, sorted unique timestamps in seconds,
                     within (0, covered duration) — canonical eval target
    boundaries     : np.ndarray int64, the same snapped to clip indices
                     (floor), unique, in [1, n_clips-1]
    labels         : np.ndarray int64 (n_clips,), verb class id or -1 (background)
    """
    fps = meta["fps"]
    n_clips = meta["n_clips"]
    delta_min = meta["delta_min"]
    clips_per_sec = fps / delta_min
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
        if len(row["verb_classes"]) == 0:
            continue  # unlabeled narration: contributes boundaries only
        covered = (clip_centers_sec >= row["start_timestamp"]) & (
            clip_centers_sec < row["end_timestamp"]
        )
        labels[covered] = int(row["verb_classes"][0])

    boundaries_sec = np.array(sorted(boundaries_sec), dtype=np.float64)
    boundaries = np.unique((boundaries_sec * clips_per_sec).astype(np.int64))
    boundaries = boundaries[(boundaries >= 1) & (boundaries <= n_clips - 1)]

    return boundaries_sec, boundaries, labels


def build_all(cfg: dict, overwrite: bool = False) -> None:
    features_dir = cfg["paths"]["features_dir"]
    narrations = load_narrations(cfg["paths"]["narrations_pkl"])
    verb_names = load_verb_names(cfg["paths"]["verb_classes_csv"])

    meta_files = sorted(
        f for f in os.listdir(features_dir) if f.endswith("_meta.json")
    )
    if not meta_files:
        raise FileNotFoundError(f"No feature meta files in {features_dir}")

    for meta_file in meta_files:
        video_id = meta_file[: -len("_meta.json")]
        out_npz = os.path.join(features_dir, video_id + "_gt.npz")
        if os.path.exists(out_npz) and not overwrite:
            logger.info(f"{video_id}: cached, skipping")
            continue

        meta = load_meta(features_dir, video_id)
        boundaries_sec, boundaries, labels = gt_for_video(narrations, video_id, meta)
        np.savez(
            out_npz,
            boundaries_sec=boundaries_sec,
            boundaries=boundaries,
            labels=labels,
        )

        n_bg = int((labels == -1).sum())
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        top = sorted(zip(counts, unique), reverse=True)[:5]
        top_str = ", ".join(
            f"{verb_names.get(int(v), f'class_{v}')}({c})" for c, v in top
        )
        logger.info(
            f"{video_id}: {meta['n_clips']} clips, "
            f"{len(boundaries_sec)} GT boundaries "
            f"({len(boundaries)} distinct clip positions), "
            f"background {n_bg}/{meta['n_clips']} clips "
            f"({100.0 * n_bg / meta['n_clips']:.0f}%), top verbs: {top_str}"
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Build GT boundaries + per-clip verb labels for all cached videos"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rebuild cached GT files")
    args = parser.parse_args()
    build_all(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
