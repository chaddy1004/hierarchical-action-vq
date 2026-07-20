"""
havq/trainer/train_lam.py
=========================
Train the NSVQ latent-action model (havq/model/vq.py) on frozen V-JEPA clip
features. LAPA-style: all (v_t, v_{t+H}) pairs are pooled from the feature cache
(pairs never cross a video boundary), the code is forced to carry the transition.

HELD-OUT-VIDEO split: whole videos are assigned to train or val, stratified by
participant, so NO clip from a val video is ever seen in training. The val video
ids are saved (checkpoint + train_log) so tokenization + BPE eval run on the same
held-out set. Adapted from archive/v1-derisk/havq/train.py (random-pair split →
held-out-video split; new feature-cache layout).

Health signals: val recon-MSE must beat the COPY baseline (v_hat = v_t, "predict
no change"), and codebook perplexity must stay >> 1 (no collapse).

Writes <results_dir>/nsvq.pt (weights + hparams + val_video_ids) and
<results_dir>/train_log.json (metric curves + the split).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from havq.model.vq import NSVQ, l2_normalize, perplexity
from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


def load_video_pairs(features_dir: str, H: int):
    """Per-video (v_t, v_{t+H}) pairs from every feature file. Returns
    (list of (video_id, Xt, XtH), D). Features are L2-normalized."""
    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files in {features_dir}")
    videos, D = [], None
    for f in npy_files:
        video_id = os.path.splitext(os.path.basename(f))[0]
        V = l2_normalize(np.load(f).astype(np.float32))
        if V.shape[0] <= H:
            logger.warning(f"{video_id}: only {V.shape[0]} clips (<= H={H}), skipping")
            continue
        videos.append((video_id, V[:-H], V[H:]))
        D = V.shape[1]
    if not videos:
        raise RuntimeError(f"No video had > H={H} clips in {features_dir}")
    return videos, D


def heldout_video_split(video_ids, val_fraction: float, seed: int):
    """Hold out a fraction of WHOLE videos per participant (video-id prefix P0x).
    Returns the set of val video ids. No clip from a val video is in train."""
    rng = np.random.default_rng(seed)
    by_part: dict[str, list[str]] = {}
    for v in video_ids:
        by_part.setdefault(v.split("-")[0], []).append(v)
    val = set()
    for _, vs in sorted(by_part.items()):
        vs = list(vs)
        rng.shuffle(vs)
        k = max(1, int(val_fraction * len(vs)))
        val.update(vs[:k])
    return val


def _stack(videos, keep: set) -> tuple[torch.Tensor, torch.Tensor]:
    xt = np.concatenate([a for vid, a, b in videos if vid in keep])
    xth = np.concatenate([b for vid, a, b in videos if vid in keep])
    return torch.from_numpy(xt), torch.from_numpy(xth)


def pick_device(pref: str) -> str:
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        logger.warning("cuda requested but unavailable; using cpu")
    return "cpu"


@torch.no_grad()
def evaluate(model: NSVQ, vt: torch.Tensor, vtH: torch.Tensor, batch_size: int):
    model.eval()
    se, n, idxs = 0.0, 0, []
    for i in range(0, vt.shape[0], batch_size):
        vhat, idx = model(vt[i:i + batch_size], vtH[i:i + batch_size])
        se += F.mse_loss(vhat, vtH[i:i + batch_size], reduction="sum").item()
        n += vhat.numel()
        idxs.append(idx)
    return se / n, perplexity(torch.cat(idxs), model.codebook_size)


def train(cfg: dict) -> None:
    tcfg, mcfg, ncfg = cfg["train"], cfg["model"], cfg["nsvq"]
    device = pick_device(tcfg["device"])
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    features_dir = os.path.join(
        cfg["paths"]["features_root"],
        feature_subdir(cfg["backbone"], cfg["scale"]["n_clip_frame"], cfg["scale"]["stride"]),
    )
    videos, D = load_video_pairs(features_dir, mcfg["H"])
    all_ids = [v for v, _, _ in videos]
    val_ids = heldout_video_split(all_ids, tcfg["val_fraction"], tcfg["seed"])
    train_ids = set(all_ids) - val_ids

    vt_tr, vtH_tr = _stack(videos, train_ids)
    vt_val, vtH_val = _stack(videos, val_ids)
    vt_tr, vtH_tr = vt_tr.to(device), vtH_tr.to(device)
    vt_val, vtH_val = vt_val.to(device), vtH_val.to(device)
    logger.info(f"device {device}, D={D}, H={mcfg['H']}")
    logger.info(f"train {len(train_ids)} videos / {vt_tr.shape[0]} pairs | "
                f"val {len(val_ids)} videos / {vt_val.shape[0]} pairs (held-out whole videos)")

    # copy baseline: predict no change (v_hat = v_t). The model must beat it.
    copy_mse = F.mse_loss(vt_val, vtH_val).item()
    logger.info(f"copy-baseline val MSE (v_hat = v_t): {copy_mse:.5f}")

    model = NSVQ(D, mcfg["latent_dim"], mcfg["codebook_size"], mcfg["enc_hidden"], mcfg["dec_hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loader = DataLoader(TensorDataset(vt_tr, vtH_tr), batch_size=tcfg["batch_size"], shuffle=True)

    history, gstep = [], 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        for vt_b, vtH_b in loader:
            vhat, _ = model(vt_b, vtH_b)
            loss = F.mse_loss(vhat, vtH_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            gstep += 1
            if gstep <= ncfg["replacement_warmup_batches"] and gstep % ncfg["replace_dead_codes_every"] == 0:
                model.replace_dead_codes()

        val_mse, ppl = evaluate(model, vt_val, vtH_val, tcfg["batch_size"])
        history.append({"epoch": epoch, "val_mse": val_mse, "perplexity": ppl})
        if epoch % 5 == 0 or epoch == tcfg["epochs"] - 1:
            logger.info(f"epoch {epoch:3d}  val_mse {val_mse:.5f}  "
                        f"(copy {copy_mse:.5f}, {100 * (1 - val_mse / copy_mse):+.1f}% vs copy)  "
                        f"perplexity {ppl:.1f}/{model.codebook_size}")

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "nsvq.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "hparams": {"in_dim": D, "latent_dim": mcfg["latent_dim"], "codebook_size": mcfg["codebook_size"],
                    "enc_hidden": mcfg["enc_hidden"], "dec_hidden": mcfg["dec_hidden"]},
        "H": mcfg["H"], "normalize": "l2", "copy_mse": copy_mse,
        "val_video_ids": sorted(val_ids),
    }, ckpt_path)
    with open(os.path.join(results_dir, "train_log.json"), "w") as f:
        json.dump({"copy_mse": copy_mse, "history": history,
                   "val_video_ids": sorted(val_ids), "train_video_ids": sorted(train_ids)}, f, indent=2)

    final = history[-1]
    logger.info(f"Saved {ckpt_path}. Final val_mse {final['val_mse']:.5f} vs copy {copy_mse:.5f} "
                f"({100 * (1 - final['val_mse'] / copy_mse):+.1f}%), perplexity {final['perplexity']:.1f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train NSVQ latent-action model (held-out-video split)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
