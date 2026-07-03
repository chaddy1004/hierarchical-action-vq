"""
havq/features.py
================
Cached V-JEPA feature extraction.

Tiles each video in `paths.videos_dir` into non-overlapping `delta_min`-frame
clips and stores one mean-pooled V-JEPA embedding per clip:

    <features_dir>/<video_id>.npy        float32 (n_clips, D), raw (un-normalized)
    <features_dir>/<video_id>_meta.json  fps, n_clips, delta_min, model id

video_id = basename of the .mp4 without extension. Extraction is one-time:
videos with an existing .npy are skipped unless --overwrite is given.

The extractor and tiling loop are copied (and adapted) from stepsegmenter's
VJepaExtractor and Segmenter._tile_and_embed; this repo stays code-independent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import glob
import av
import numpy as np
import torch

from havq import load_config

from transformers import AutoModel, AutoVideoProcessor

logger = logging.getLogger(__name__)


class VJepaExtractor:
    """V-JEPA2 feature extractor using HuggingFace AutoModel."""

    def __init__(self, model_path: str, device: str = "cuda"):


        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        logger.info(f"V-JEPA extractor loaded from {model_path}")

    def extract_feature(self, video: np.ndarray) -> np.ndarray:
        """
        Args:
            video: T x H x W x C uint8 numpy array
        Returns:
            mean-pooled clip feature as numpy array, shape (D,)
        """
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        inputs = self.processor(video_tensor, return_tensors="pt")["pixel_values_videos"]
        inputs = inputs.to(self.device)
        del video_tensor
        with torch.inference_mode():
            features = self.model.get_vision_features(inputs)
        del inputs
        emb = features.cpu().numpy()
        del features
        if emb.ndim > 1:
            emb = emb.mean(axis=tuple(range(emb.ndim - 1)))
        return emb


def tile_and_embed(
    video_path: str, extractor: VJepaExtractor, delta_min: int
) -> tuple[np.ndarray, float]:
    """Stream video, embed each delta_min-frame clip on the fly.

    Returns
    -------
    V : np.ndarray  float32 (n_clips, D)
        One embedding per atomic clip. A trailing partial clip (< delta_min
        frames) is dropped.
    fps : float
    """
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)

    embeddings = []
    clip_buffer = []

    for frame in container.decode(video=0):
        clip_buffer.append(frame.to_ndarray(format="rgb24"))
        if len(clip_buffer) == delta_min:
            clip = np.stack(clip_buffer, axis=0)
            print(f"  embedding clip {len(embeddings) + 1}...", end="\r")
            embeddings.append(extractor.extract_feature(clip))
            del clip
            clip_buffer = []

    container.close()
    print(f"\n  {len(embeddings)} atomic clips")
    return np.stack(embeddings, axis=0).astype(np.float32), fps


def extract_all(cfg: dict, overwrite: bool = False) -> None:
    videos_dir = cfg["paths"]["videos_dir"]
    features_dir = cfg["paths"]["features_dir"]
    delta_min = cfg["features"]["delta_min"]
    model_path = cfg["features"]["vjepa_model"]

    os.makedirs(features_dir, exist_ok=True)


    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")

    pending = []
    for video_path in video_files:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_npy = os.path.join(features_dir, video_id + ".npy")
        if os.path.exists(out_npy) and not overwrite:
            logger.info(f"{video_id}: cached, skipping")
            continue
        pending.append((video_path, video_id, out_npy))

    if not pending:
        logger.info("All videos cached, nothing to do")
        return

    extractor = VJepaExtractor(model_path, device=cfg["train"]["device"])

    for video_path, video_id, out_npy in pending:
        logger.info(f"Extracting {video_id}")
        V, fps = tile_and_embed(video_path, extractor, delta_min)
        np.save(out_npy, V)
        meta = {
            "video_id": video_id,
            "fps": fps,
            "n_clips": int(V.shape[0]),
            "feature_dim": int(V.shape[1]),
            "delta_min": delta_min,
            "vjepa_model": model_path,
        }
        with open(os.path.join(features_dir, video_id + "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"{video_id}: saved {V.shape} to {out_npy}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Cache V-JEPA clip features for all videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="re-extract cached videos")
    args = parser.parse_args()
    extract_all(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
