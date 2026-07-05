"""
havq/data/preprocessing/preprocessing.py
========================================
Cached clip feature extraction. Driven by the standalone preprocessing config
(havq/data/preprocessing/configs/*.yaml): flat `model` / `model_weight` /
`n_clip_frame` / `stride` keys plus `paths.{videos_dir, features_dir}` and
`device`.

Walks each video in `paths.videos_dir` with a sliding window of `n_clip_frame`
frames advancing by `stride` frames and stores one pooled embedding per clip:

    <features_dir>/<model>_clip<n>_stride<s>/<video_id>.npy        float32 (n_clips, D)
    <features_dir>/<model>_clip<n>_stride<s>/<video_id>_meta.json  fps, n_clips, ...

stride == n_clip_frame  -> non-overlapping tiling
stride <  n_clip_frame  -> overlapping windows
stride >  n_clip_frame  -> gapped subsampling

Clip i starts at frame i * stride, i.e. at second (i * stride) / fps. Outputs
are grouped in one subdirectory per (model, n_clip_frame, stride) setting (see
havq.utils.paths.feature_subdir) so multiple backbones / temporal scales cache
side by side, and training can reconstruct the exact path. A trailing partial
clip (< n_clip_frame frames) is dropped. video_id = basename of the .mp4
without extension. Extraction is one-time: videos with an existing .npy are
skipped unless overwrite is given.

Videos are NEVER loaded whole: frames are decoded one at a time (PyAV) into a
buffer holding at most n_clip_frame frames, so memory stays flat regardless of
video length.
"""

from __future__ import annotations

from glob import glob
import json
import logging
import os

import av
import numpy as np

from havq.data.preprocessing.extractors.base_extractor import BaseVideoExtractor
from havq.data.preprocessing.extractors.vjepa_extractor import VJepaExtractor
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)



def video_fps(video_path: str) -> float:
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)
    container.close()
    return fps


def stream_clips(video_path: str, n_clip_frame: int, stride: int):
    """Yield (start_frame, clip) for each full window, decoding frame by frame.

    clip is a (n_clip_frame, H, W, 3) uint8 array; start_frame = i * stride for
    the i-th clip. At most n_clip_frame frames are held in memory. A trailing
    window with fewer than n_clip_frame frames is dropped.
    """
    container = av.open(video_path)
    try:
        buf = []
        next_start = 0
        for idx, frame in enumerate(container.decode(video=0)):
            if idx < next_start:
                continue  # gap between windows when stride > n_clip_frame
            buf.append(frame.to_ndarray(format="rgb24"))
            if len(buf) == n_clip_frame:
                yield next_start, np.stack(buf, axis=0)
                next_start += stride
                # overlapping windows share the tail of the buffer
                buf = buf[stride:] if stride < n_clip_frame else []
    finally:
        container.close()


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self._extractor: BaseVideoExtractor | None = None

    @property
    def extractor(self) -> BaseVideoExtractor:
        """Lazily build the backbone named by `config["model"]` (factory)."""
        if self._extractor is None:
            model = self.config["model"]
            if model == "vjepa":
                self._extractor = VJepaExtractor(
                    self.config["model_weight"], self.config["device"]
                )
            else:
                raise ValueError(f"Unknown extractor model: {model!r}")
        return self._extractor

    def feature_extraction(self, overwrite: bool = False) -> str:
        """Extract and cache clip features for every video. All settings come
        from the config; returns the output directory."""
        model = self.config["model"]
        n_clip_frame = self.config["n_clip_frame"]
        stride = self.config["stride"]
        videos_dir = self.config["paths"]["videos_dir"]
        features_dir = self.config["paths"]["features_dir"]

        out_dir = os.path.join(features_dir, feature_subdir(model, n_clip_frame, stride))
        os.makedirs(out_dir, exist_ok=True)

        video_paths = sorted(glob(os.path.join(videos_dir, "*", "*.mp4")))
        if not video_paths:
            raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")

        pending = []
        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            out_npy = os.path.join(out_dir, video_id + ".npy")
            if os.path.exists(out_npy) and not overwrite:
                logger.info(f"{video_id}: cached, skipping")
                continue
            pending.append((video_path, video_id, out_npy))

        for video_path, video_id, out_npy in pending:
            logger.info(f"Extracting {video_id} (n_clip_frame={n_clip_frame}, stride={stride})")
            embeddings = []
            for start, clip in stream_clips(video_path, n_clip_frame, stride):
                print(f"  embedding clip {len(embeddings) + 1} (frame {start})...", end="\r")
                embeddings.append(self.extractor.extract_feature(clip))
            print()
            if not embeddings:
                logger.warning(f"{video_id}: shorter than {n_clip_frame} frames, skipping")
                continue

            V = np.stack(embeddings, axis=0).astype(np.float32)
            np.save(out_npy, V)
            meta = {
                "video_id": video_id,
                "fps": video_fps(video_path),
                "n_clips": int(V.shape[0]),
                "feature_dim": int(V.shape[1]),
                "model": model,
                "model_weight": self.config["model_weight"],
                "n_clip_frame": n_clip_frame,
                "stride": stride,
            }
            with open(os.path.join(out_dir, video_id + "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"{video_id}: saved {V.shape} to {out_npy}")

        return out_dir
