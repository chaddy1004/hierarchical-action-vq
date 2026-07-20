"""
havq/data/preprocessing/lapa_tokenize.py
========================================
HD-EPIC video -> one LAPA latent action per frame pair, cached to disk.

Collapses stages (1) and (3) of CODE_MAP.md into a single inference pass: LAPA's
LAQ is already a trained latent-action tokenizer, so there is no feature-then-train
split and nothing here learns anything.

PAIRING. Frames are sampled every `pair_gap` frames and paired back-to-back:

    (f_0, f_g), (f_g, f_2g), (f_2g, f_3g), ...        g = pair_gap

so pair j spans frames [j*g, (j+1)*g) -- exactly the clip convention with
n_clip_frame = stride = g. That is deliberate: it makes clips_per_sec = fps / g,
which is the formula havq/analysis/gt.py and havq/analysis/bpe_eval.py already
use, so both read this cache with no changes. Each sampled frame is preprocessed
ONCE and used twice (as the 'after' of pair j and the 'before' of pair j+1).

Writes, per video, into <features_dir>/lapa_clip<g>_stride<g>/ :

    <video_id>_emb.npy    float32 (n_pairs, 128)   pre-quantization delta
    <video_id>_codes.npy  int64   (n_pairs, 4)     LAPA's own codebook indices
    <video_id>_meta.json  video_id, fps, n_clips, n_clip_frame, stride, ...

`_emb.npy` is the payload -- the continuous latent action that cluster_actions.py
partitions into the K-symbol alphabet BPE consumes. `_codes.npy` is LAPA's own
12-bit quantization, cached for reference/diagnostics only; nothing downstream
requires it. `n_clips` in the meta is the pair count (named for gt.py's benefit).

Extraction is one-time: a video whose `_emb.npy` exists is skipped unless
overwrite is given. Videos are NEVER loaded whole -- frames are decoded one at a
time (PyAV), only every g-th is kept, and at most `batch_size`+1 preprocessed
frames are held at once, so memory is flat in video length.

Also logs, per video, a lag-1 cosine of consecutive deltas against a shuffled
control. This is calibration, not a verdict: it measures the timescale on which
the latent-action series decorrelates, which is what bounds a sensible pair_gap
and alphabet size K downstream.
"""

from __future__ import annotations

from glob import glob
import json
import logging
import os

import av
import numpy as np
import torch

from havq.data.preprocessing.extractors.lapa_extractor import LapaExtractor
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)

MODEL_NAME = "lapa"


def video_fps(video_path: str) -> float:
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)
    container.close()
    return fps


def stream_sampled_frames(video_path: str, pair_gap: int):
    """Yield every pair_gap-th frame (H x W x 3 uint8), decoding one at a time.

    Decoding is sequential rather than seeking: HD-EPIC .mp4s are long-GOP, so
    seeking to each keyframe-misaligned index costs more than decoding straight
    through and dropping frames we do not want.
    """
    container = av.open(video_path)
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % pair_gap == 0:
                yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def lag1_cosine(embeddings: np.ndarray, rng) -> tuple[float, float]:
    """(observed, shuffled) mean cosine between consecutive latent actions.

    `observed` is the mean cosine of (e_t, e_{t+1}). `shuffled` is the same
    statistic after permuting the rows, which preserves the marginal distribution
    of the deltas exactly and destroys only their time order -- so it is the
    matched yardstick for "how similar would consecutive vectors look by chance".

    Embeddings are MEAN-CENTERED first, and that is not cosmetic. The raw deltas
    carry a large constant offset (||mean|| is ~90% of a typical row norm, and the
    per-video means are mutually cosine ~0.99, so it is a fixed artifact of the LAQ
    encoder, not scene content). Uncentered, every delta looks similar to every
    other one -- both numbers land near +0.85 and the comparison measures the
    offset rather than the action. Centering is per-video here so the diagnostic
    stays self-contained; corpus-level centering is cluster_actions.py's job.
    """
    if len(embeddings) < 3:
        return 0.0, 0.0
    centered = embeddings - embeddings.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = centered / norms

    observed = float((unit[:-1] * unit[1:]).sum(axis=1).mean())
    permuted = unit[rng.permutation(len(unit))]
    shuffled = float((permuted[:-1] * permuted[1:]).sum(axis=1).mean())
    return observed, shuffled


def tokenize_video(extractor: LapaExtractor, video_path: str, pair_gap: int, batch_size: int):
    """One video -> (codes, embeddings) for every back-to-back frame pair.

    codes:      int64   (n_pairs, code_seq_len)
    embeddings: float32 (n_pairs, embedding_dim)
    """
    codes_chunks = []
    embedding_chunks = []
    frames = []

    for frame in stream_sampled_frames(video_path, pair_gap):
        frames.append(extractor.preprocess_frame(frame))
        # batch_size + 1 frames yield batch_size back-to-back pairs
        if len(frames) == batch_size + 1:
            codes, embeddings = extractor.extract(
                torch.stack(frames[:-1]), torch.stack(frames[1:])
            )
            codes_chunks.append(codes)
            embedding_chunks.append(embeddings)
            # carry the last frame over: it is the 'before' of the next pair
            frames = [frames[-1]]
            n_done = sum(len(c) for c in codes_chunks)
            print(f"  {n_done} pairs...", end="\r")

    if len(frames) >= 2:
        codes, embeddings = extractor.extract(
            torch.stack(frames[:-1]), torch.stack(frames[1:])
        )
        codes_chunks.append(codes)
        embedding_chunks.append(embeddings)
    print()

    if not codes_chunks:
        return None, None
    return np.concatenate(codes_chunks, axis=0), np.concatenate(embedding_chunks, axis=0)


def tokenize_all(config: dict, overwrite: bool = False, video_ids: list[str] | None = None) -> str:
    """Tokenize every video named by the config. Returns the output directory."""
    pair_gap = config["pair_gap"]
    videos_dir = config["paths"]["videos_dir"]
    features_dir = config["paths"]["features_dir"]

    # pairs are back-to-back, so the clip length and the stride are both pair_gap
    out_dir = os.path.join(features_dir, feature_subdir(MODEL_NAME, pair_gap, pair_gap))
    os.makedirs(out_dir, exist_ok=True)

    video_paths = sorted(glob(os.path.join(videos_dir, "*", "*.mp4")))
    video_paths += sorted(glob(os.path.join(videos_dir, "*.mp4")))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")

    if video_ids is not None:
        by_id = {os.path.splitext(os.path.basename(p))[0]: p for p in video_paths}
        missing = [v for v in video_ids if v not in by_id]
        if missing:
            raise FileNotFoundError(f"video_ids not found under {videos_dir}: {missing}")
        video_paths = [by_id[v] for v in video_ids]

    pending = []
    for video_path in video_paths:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_emb = os.path.join(out_dir, video_id + "_emb.npy")
        if os.path.exists(out_emb) and not overwrite:
            logger.info(f"{video_id}: cached, skipping")
            continue
        pending.append((video_path, video_id))

    extractor = LapaExtractor(config["checkpoint"], config["device"])
    rng = np.random.default_rng(config["seed"])

    for video_path, video_id in pending:
        logger.info(f"Tokenizing {video_id} (pair_gap={pair_gap})")
        codes, embeddings = tokenize_video(
            extractor, video_path, pair_gap, config["batch_size"]
        )
        if codes is None:
            logger.warning(f"{video_id}: fewer than 2 sampled frames, skipping")
            continue

        np.save(os.path.join(out_dir, video_id + "_emb.npy"), embeddings)
        np.save(os.path.join(out_dir, video_id + "_codes.npy"), codes)
        meta = {
            "video_id": video_id,
            "fps": video_fps(video_path),
            "n_clips": int(codes.shape[0]),
            "n_clip_frame": pair_gap,
            "stride": pair_gap,
            "model": MODEL_NAME,
            "checkpoint": config["checkpoint"],
            "embedding_dim": int(embeddings.shape[1]),
            "code_seq_len": extractor.code_seq_len,
            "codebook_size": extractor.codebook_size,
        }
        with open(os.path.join(out_dir, video_id + "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        observed, shuffled = lag1_cosine(embeddings, rng)
        n_distinct = len(np.unique(codes, axis=0))
        logger.info(
            f"{video_id}: {codes.shape[0]} pairs | lag-1 cos {observed:+.3f} "
            f"(shuffled {shuffled:+.3f}) | {n_distinct} distinct LAPA codes"
        )

    return out_dir
