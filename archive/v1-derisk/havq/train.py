"""
havq/train.py
=============
Train the NSVQ latent-action model (havq/vq.py) on frozen V-JEPA features.

All (v_t, v_{t+H}) pairs are pooled across every <video_id>.npy in
`paths.features_dir` (pairs never cross a video boundary). Clip embeddings are
L2-normalized so reconstruction MSE lives on the unit sphere and is directly
comparable to the copy baseline. A random `train.val_fraction` of pairs is held
out for recon-MSE / perplexity reporting.

Sanity signal (design doc §9): the trained model's val recon MSE must beat the
COPY baseline (v_hat = v_t, i.e. "predict no change"), and codebook perplexity
must stay >> 1 (no collapse). Otherwise the tokens carry nothing transition-like.

Writes `<results_dir>/nsvq.pt` (weights + the hyperparameters needed to rebuild
the model for tokenization) and `<results_dir>/train_log.json` (metric curves).
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

from havq import load_config
from havq.vq import NSVQ, l2_normalize, perplexity

logger = logging.getLogger(__name__)


def load_pairs(features_dir: str, H: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pool (v_t, v_{t+H}) pairs from every feature file. Returns (Xt, XtH, D)."""
    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files in {features_dir} — run havq/features.py first")

    Xt_list, XtH_list = [], []
    for f in npy_files:
        V = l2_normalize(np.load(f).astype(np.float32))
        if V.shape[0] <= H:
            logger.warning(f"{os.path.basename(f)}: only {V.shape[0]} clips (<= H={H}), skipping")
            continue
        Xt_list.append(V[:-H])
        XtH_list.append(V[H:])
        logger.info(f"{os.path.basename(f)}: {V.shape[0]} clips -> {V.shape[0] - H} pairs")

    Xt = np.concatenate(Xt_list, axis=0)
    XtH = np.concatenate(XtH_list, axis=0)
    return torch.from_numpy(Xt), torch.from_numpy(XtH), Xt.shape[1]


@torch.no_grad()
def evaluate(model: NSVQ, vt: torch.Tensor, vtH: torch.Tensor, batch_size: int) -> tuple[float, float]:
    """Val recon MSE (hard-quantized) and codebook perplexity over all val pairs."""
    model.eval()
    se, n, idxs = 0.0, 0, []
    for i in range(0, vt.shape[0], batch_size):
        vhat, idx = model(vt[i : i + batch_size], vtH[i : i + batch_size])
        se += F.mse_loss(vhat, vtH[i : i + batch_size], reduction="sum").item()
        n += vhat.numel()
        idxs.append(idx)
    mse = se / n
    ppl = perplexity(torch.cat(idxs), model.codebook_size)
    return mse, ppl


def train(cfg: dict) -> None:
    tcfg, mcfg, ncfg = cfg["train"], cfg["model"], cfg["nsvq"]
    device = tcfg["device"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    Xt, XtH, D = load_pairs(cfg["paths"]["features_dir"], mcfg["H"])
    logger.info(f"Total {Xt.shape[0]} pairs, feature dim D={D}")

    # deterministic train/val split
    g = torch.Generator().manual_seed(tcfg["seed"])
    perm = torch.randperm(Xt.shape[0], generator=g)
    n_val = int(tcfg["val_fraction"] * Xt.shape[0])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    vt_tr, vtH_tr = Xt[tr_idx].to(device), XtH[tr_idx].to(device)
    vt_val, vtH_val = Xt[val_idx].to(device), XtH[val_idx].to(device)
    logger.info(f"Train {vt_tr.shape[0]} / Val {vt_val.shape[0]} pairs")

    # copy baseline: predict no change (v_hat = v_t). Constant, model must beat it.
    copy_mse = F.mse_loss(vt_val, vtH_val).item()
    logger.info(f"Copy-baseline val MSE (v_hat = v_t): {copy_mse:.5f}")

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
                n_dead = model.replace_dead_codes()
                logger.info(f"  [batch {gstep}] replaced {n_dead} dead codes")

        val_mse, ppl = evaluate(model, vt_val, vtH_val, tcfg["batch_size"])
        history.append({"epoch": epoch, "val_mse": val_mse, "perplexity": ppl})
        if epoch % 10 == 0 or epoch == tcfg["epochs"] - 1:
            logger.info(
                f"epoch {epoch:3d}  val_mse {val_mse:.5f}  "
                f"(copy {copy_mse:.5f}, {100 * (1 - val_mse / copy_mse):+.1f}% vs copy)  "
                f"perplexity {ppl:.1f}/{model.codebook_size}"
            )

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["results_dir"], "nsvq.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": {
                "in_dim": D,
                "latent_dim": mcfg["latent_dim"],
                "codebook_size": mcfg["codebook_size"],
                "enc_hidden": mcfg["enc_hidden"],
                "dec_hidden": mcfg["dec_hidden"],
            },
            "H": mcfg["H"],
            "normalize": "l2",
            "copy_mse": copy_mse,
        },
        ckpt_path,
    )
    with open(os.path.join(cfg["paths"]["results_dir"], "train_log.json"), "w") as f:
        json.dump({"copy_mse": copy_mse, "history": history}, f, indent=2)

    final = history[-1]
    logger.info(
        f"Saved {ckpt_path}. Final val_mse {final['val_mse']:.5f} vs copy {copy_mse:.5f} "
        f"({100 * (1 - final['val_mse'] / copy_mse):+.1f}%), perplexity {final['perplexity']:.1f}"
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train NSVQ latent-action model on V-JEPA features")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
