"""Split video_ids into N load-balanced shard files for parallel/cluster extraction.

Videos are assigned via greedy LPT (longest-processing-time-first): sorted by
duration descending, each video goes to whichever shard currently has the
smallest total duration. This keeps shard wall-clock time roughly equal even
though HD-EPIC video lengths vary a lot -- a plain contiguous or interleaved
split does not guarantee that.

Usage:
    python make_shards.py --videos-dir /path/to/HD-EPIC/Videos --n-shards 8 --out-dir shards

Each shard is written to <out-dir>/shard_<i>.txt (one video_id per line). Shards
are disjoint, so N processes/jobs can each own one shard file with no race
condition -- every video is claimed by exactly one shard, decided up front. See
run_preprocessing.py --shard-file.
"""

from __future__ import annotations

import argparse
from glob import glob
import os

import av


def video_duration(video_path: str) -> float:
    """Duration in seconds, read from container metadata (no frame decode)."""
    container = av.open(video_path)
    duration = float(container.duration) / av.time_base
    container.close()
    return duration


def lpt_shards(video_ids_and_durations: list[tuple[str, float]], n_shards: int):
    """Greedy longest-processing-time-first: biggest videos placed first, each
    into the currently-lightest shard. Keeps shard totals close to equal."""
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    totals = [0.0] * n_shards
    for video_id, duration in sorted(video_ids_and_durations, key=lambda x: x[1], reverse=True):
        i = totals.index(min(totals))
        shards[i].append(video_id)
        totals[i] += duration
    return shards, totals


def main():
    ap = argparse.ArgumentParser(description="Split dataset video_ids into N load-balanced shards")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", default="shards")
    args = ap.parse_args()

    video_paths = sorted(glob(os.path.join(args.videos_dir, "*", "*.mp4")))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {args.videos_dir}")

    print(f"Reading duration metadata for {len(video_paths)} videos...")
    video_ids_and_durations = []
    for video_path in video_paths:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        video_ids_and_durations.append((video_id, video_duration(video_path)))

    shards, totals = lpt_shards(video_ids_and_durations, args.n_shards)

    os.makedirs(args.out_dir, exist_ok=True)
    n_digits = len(str(args.n_shards - 1))
    for i, shard in enumerate(shards):
        shard_path = os.path.join(args.out_dir, f"shard_{i:0{n_digits}d}.txt")
        with open(shard_path, "w") as f:
            f.write("\n".join(shard) + "\n")
        print(f"{shard_path}: {len(shard)} videos, {totals[i] / 3600:.2f}h")

    spread = (max(totals) - min(totals)) / 3600
    print(f"{len(video_ids_and_durations)} videos split into {args.n_shards} shards "
          f"in {args.out_dir}/ (max-min spread: {spread:.2f}h)")


if __name__ == "__main__":
    main()
