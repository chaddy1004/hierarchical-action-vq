"""
hicap/data/vjepa_extract.py
===========================
Extract V-JEPA 2 clip features from RGB videos, for the "better backbone than I3D"
experiment. Produces per-clip feature sequences that drop into the same hierarchy
+ eval pipeline as the I3D features (hicap/eval/{gate,compare}.py), so V-JEPA vs
I3D is a one-variable swap.

DESIGN (the clip-length question). V-JEPA vitg-fpc64 embeds a 64-frame window into
one vector. We slide that window with `stride` frames:

    clip j covers frames [j*stride, j*stride + window),  one 1408-d vector per clip

so the feature sequence has length ~T/stride at the video's frame rate. `window`
sets how much time one embedding pools over (64 frames = 4.3 s @15 fps / 2.1 s
@30 fps); `stride` sets the token rate. Both are config, so the temporal scale is
explicit and tunable -- pick a stride that gives a sensible number of tokens vs the
annotation granularity (e.g. stride 8-16 for Breakfast's ~10 actions).

Because features are at clip resolution but the GT is per-frame, evaluation must
align them: hicap.data.tas.load_labels_strided downsamples the GT to the window
centers using the meta written here. So the SAME gate/compare run on V-JEPA
features by pointing --data-root at the V-JEPA output and passing the stride/window.

Writes, per video, into <out_dir>/<dataset>/:
    features/<video_id>.npy   float32 (n_clips, 1408)  row-major (T, D)
    features/<video_id>.json  {fps, window, stride, n_clips, n_frames, model}

Backbone code adapted from archive/v2-vjepa-lapa/havq (VJepaExtractor + PyAV
streaming); copied, not imported, per STYLE.md. Videos are never loaded whole --
frames stream one at a time with a bounded window buffer.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)


class VJepaExtractor:
    """V-JEPA 2 feature extractor (HF AutoModel.get_vision_features, mean-pooled)."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        logger.info(f"V-JEPA loaded from {model_path}")

    @torch.no_grad()
    def embed_clip(self, clip):
        """clip: (F, H, W, 3) uint8 -> (D,) float32 mean-pooled feature."""
        video = torch.from_numpy(clip).permute(0, 3, 1, 2)  # F,C,H,W
        inputs = self.processor(video, return_tensors="pt")["pixel_values_videos"].to(self.device)
        feats = self.model.get_vision_features(inputs).cpu().numpy()
        if feats.ndim > 1:
            feats = feats.mean(axis=tuple(range(feats.ndim - 1)))
        return feats.astype(np.float32)


def video_fps(video_path):
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)
    container.close()
    return fps


def stream_windows(video_path, window, stride):
    """Yield (start_frame, clip) for each full window; frame-by-frame decode.

    clip is (window, H, W, 3) uint8; at most `window` frames held in memory. A
    trailing partial window is dropped.
    """
    container = av.open(video_path)
    try:
        buf = []
        next_start = 0
        for idx, frame in enumerate(container.decode(video=0)):
            if idx < next_start:
                continue
            buf.append(frame.to_ndarray(format="rgb24"))
            if len(buf) == window:
                yield next_start, np.stack(buf, axis=0)
                next_start += stride
                buf = buf[stride:] if stride < window else []
    finally:
        container.close()


def extract_video(extractor, video_path, window, stride):
    """One video -> (n_clips, D) features + (fps, n_frames)."""
    feats = []
    n_frames = 0
    for start, clip in stream_windows(video_path, window, stride):
        feats.append(extractor.embed_clip(clip))
        n_frames = start + window
        print(f"  clip {len(feats)} (frame {start})...", end="\r")
    print()
    if not feats:
        return None, 0
    return np.stack(feats, axis=0), n_frames


def extract_all(cfg):
    out_dir = os.path.join(cfg["paths"]["out_root"], cfg["dataset"], "features")
    os.makedirs(out_dir, exist_ok=True)
    patterns = [os.path.join(cfg["paths"]["rgb_root"], p) for p in cfg["rgb_globs"]]
    video_paths = sorted(set(sum([glob(p) for p in patterns], [])))
    if not video_paths:
        raise FileNotFoundError(f"No videos under {cfg['paths']['rgb_root']} matching {cfg['rgb_globs']}")
    logger.info(f"{len(video_paths)} videos | window={cfg['window']} stride={cfg['stride']}")

    extractor = VJepaExtractor(cfg["model_path"], cfg["device"])
    for i, vp in enumerate(video_paths):
        vid = os.path.splitext(os.path.basename(vp))[0]
        out_npy = os.path.join(out_dir, vid + ".npy")
        if os.path.exists(out_npy) and not cfg.get("overwrite"):
            logger.info(f"[{i + 1}/{len(video_paths)}] {vid}: cached")
            continue
        feats, n_frames = extract_video(extractor, vp, cfg["window"], cfg["stride"])
        if feats is None:
            logger.warning(f"{vid}: shorter than {cfg['window']} frames, skipped")
            continue
        np.save(out_npy, feats)
        meta = {
            "video_id": vid, "fps": video_fps(vp), "window": cfg["window"],
            "stride": cfg["stride"], "n_clips": int(feats.shape[0]),
            "n_frames": int(n_frames), "model": os.path.basename(cfg["model_path"]),
        }
        with open(os.path.join(out_dir, vid + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[{i + 1}/{len(video_paths)}] {vid}: {feats.shape} -> {out_npy}")
    return out_dir
