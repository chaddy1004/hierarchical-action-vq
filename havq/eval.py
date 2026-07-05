"""
havq/eval.py
============
§9 de-risking analysis: do the discrete latent-action tokens align with real
action structure? The PRIMARY signal is boundary alignment; per-clip
action-meaningfulness (NMI / purity) is reported as a secondary check.

For each video with both a token stream (havq/tokenize_videos.py) and GT
(havq/gt.py):

  Predicted boundaries -- clip positions where the token changes -- are mapped to
  seconds (clip j starts at j / clips_per_sec) and matched one-to-one, within a
  tolerance, against the CONTINUOUS GT boundaries (boundaries_sec). We report
  boundary precision / recall / F1 at several tolerances.

A raw F1 means nothing without a yardstick, so each is compared against two
same-budget baselines (identical number of predicted boundaries):

  random  : boundaries at random clip positions (averaged over many trials)
  change  : the stepsegmenter-style cue -- top-K peaks of the V-JEPA embedding
            change score 1 - cos(v_j, v_{j-1}). Beating THIS is the real test:
            it shows the VQ tokens capture transitions beyond raw feature change.

Outputs:
  <results_dir>/eval_report.json        all metrics, per video + aggregate
  <results_dir>/plots/<video_id>.png    token stream + GT vs predicted boundaries
  console summary with the §9 decision signals
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

from havq import load_config  # noqa: E402
from havq.gt import load_meta  # noqa: E402
from havq.vq import l2_normalize  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- boundary F1

def match_count(pred_sec: np.ndarray, gt_sec: np.ndarray, tol: float) -> int:
    """Greedy one-to-one matches between predicted and GT boundaries within tol."""
    if len(pred_sec) == 0 or len(gt_sec) == 0:
        return 0
    pairs = []
    for i, p in enumerate(pred_sec):
        for j, g in enumerate(gt_sec):
            d = abs(p - g)
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": n_pred, "n_gt": n_gt}


def boundary_f1(pred_sec: np.ndarray, gt_sec: np.ndarray, tol: float) -> dict:
    return prf(match_count(pred_sec, gt_sec, tol), len(pred_sec), len(gt_sec))


# ----------------------------------------------------------------- baselines

def random_f1(n_pred: int, n_positions: int, gt_sec: np.ndarray, clips_per_sec: float,
              tol: float, trials: int, seed: int) -> float:
    """Mean F1 of n_pred boundaries placed at random clip positions in [1, n_positions)."""
    if n_pred == 0 or n_positions <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    k = min(n_pred, n_positions - 1)
    f1s = []
    for _ in range(trials):
        clips = rng.choice(np.arange(1, n_positions), size=k, replace=False)
        f1s.append(boundary_f1(clips / clips_per_sec, gt_sec, tol)["f1"])
    return float(np.mean(f1s))


def change_boundaries(V_norm: np.ndarray, k: int) -> np.ndarray:
    """Top-k clip positions by V-JEPA embedding change score 1 - cos(v_j, v_{j-1})."""
    cos = (V_norm[1:] * V_norm[:-1]).sum(axis=1)  # normalized -> dot = cosine
    change = 1.0 - cos  # change[j-1] scores the boundary at clip j
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    top = np.argsort(change)[::-1][:k]
    return np.sort(top + 1)  # clip index of the boundary


# ---------------------------------------------- action-meaningfulness (2ndary)

def purity(tokens: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    t, l = tokens[mask], labels[mask]
    if len(t) == 0:
        return float("nan")
    total = sum(int(np.bincount(l[t == tok]).max()) for tok in np.unique(t))
    return total / len(t)


def nmi(tokens: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    t, l = tokens[mask], labels[mask]
    if len(t) < 2 or len(np.unique(t)) < 2 or len(np.unique(l)) < 2:
        return float("nan")
    return float(normalized_mutual_info_score(l, t))


# ----------------------------------------------------------------- per video

def plot_video(video_id, tokens, pred_clips, gt_sec, clips_per_sec, out_png):
    n = len(tokens)
    t_clips = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(14, 4), height_ratios=[3, 1], sharex=True)
    ax[0].scatter(t_clips, tokens, s=6, c=tokens, cmap="tab20", linewidths=0)
    ax[0].set_ylabel("token id")
    ax[0].set_title(f"{video_id}: token stream (top) & boundaries (bottom)")
    for c in pred_clips:
        ax[1].axvline(c, color="tab:red", lw=0.8, alpha=0.7)
    for s in gt_sec * clips_per_sec:
        ax[1].axvline(s, color="tab:blue", lw=0.8, alpha=0.5, ls="--")
    ax[1].set_yticks([])
    ax[1].set_xlabel("clip index")
    ax[1].plot([], [], color="tab:red", label="predicted (token change)")
    ax[1].plot([], [], color="tab:blue", ls="--", label="GT boundary")
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def eval_video(video_id, features_dir, tokens_dir, ecfg, seed) -> dict | None:
    tok_path = os.path.join(tokens_dir, video_id + ".npy")
    gt_path = os.path.join(features_dir, video_id + "_gt.npz")
    feat_path = os.path.join(features_dir, video_id + ".npy")
    if not (os.path.exists(tok_path) and os.path.exists(gt_path)):
        return None

    tokens = np.load(tok_path)
    gt = np.load(gt_path)
    gt_sec = gt["boundaries_sec"]
    labels = gt["labels"]
    meta = load_meta(features_dir, video_id)
    clips_per_sec = meta["fps"] / meta["delta_min"]

    pred_clips = np.flatnonzero(np.diff(tokens)) + 1  # token-change positions
    pred_sec = pred_clips / clips_per_sec
    k = len(pred_clips)

    V_norm = l2_normalize(np.load(feat_path).astype(np.float32))
    chg_clips = change_boundaries(V_norm, k)
    chg_sec = chg_clips / clips_per_sec

    tol_p = ecfg["primary_tolerance_sec"]
    res = {
        "n_clips": int(meta["n_clips"]),
        "n_tokens": int(len(tokens)),
        "n_gt": int(len(gt_sec)),
        "n_pred": int(k),
        "model_f1": {str(t): boundary_f1(pred_sec, gt_sec, t) for t in ecfg["tolerances_sec"]},
        "random_f1_primary": random_f1(k, len(tokens), gt_sec, clips_per_sec, tol_p,
                                       ecfg["random_trials"], seed),
        "change_f1_primary": boundary_f1(chg_sec, gt_sec, tol_p)["f1"],
        "nmi": nmi(tokens, labels[: len(tokens)]),
        "purity": purity(tokens, labels[: len(tokens)]),
    }
    return res, (video_id, tokens, pred_clips, gt_sec, clips_per_sec)


# ------------------------------------------------------------------- driver

def evaluate_all(cfg: dict) -> None:
    features_dir = cfg["paths"]["features_dir"]
    tokens_dir = cfg["paths"]["tokens_dir"]
    results_dir = cfg["paths"]["results_dir"]
    ecfg = cfg["eval"]
    seed = cfg["train"]["seed"]
    tol_p = ecfg["primary_tolerance_sec"]
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    video_ids = sorted(
        os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(tokens_dir, "*.npy"))
    )
    if not video_ids:
        raise FileNotFoundError(f"No token streams in {tokens_dir} — run havq/tokenize_videos.py first")

    per_video, agg_tp, agg_pred, agg_gt = {}, 0, 0, 0
    for vid in video_ids:
        out = eval_video(vid, features_dir, tokens_dir, ecfg, seed)
        if out is None:
            logger.warning(f"{vid}: no GT (skipping — not an annotated video?)")
            continue
        res, plot_args = out
        per_video[vid] = res
        try:
            plot_video(*plot_args, os.path.join(plots_dir, vid + ".png"))
        except Exception as e:  # a plotting hiccup must not sink the numeric eval
            logger.warning(f"{vid}: plot failed ({e})")

        prim = res["model_f1"][str(tol_p)]
        agg_tp += prim["tp"]
        agg_pred += prim["n_pred"]
        agg_gt += prim["n_gt"]
        logger.info(
            f"{vid}: F1@{tol_p}s model {prim['f1']:.3f} "
            f"(P {prim['precision']:.3f} R {prim['recall']:.3f}) | "
            f"change {res['change_f1_primary']:.3f} | random {res['random_f1_primary']:.3f} | "
            f"NMI {res['nmi']:.3f} purity {res['purity']:.3f} "
            f"[{res['n_pred']} pred vs {res['n_gt']} GT boundaries]"
        )

    micro = prf(agg_tp, agg_pred, agg_gt)

    # copy-baseline / perplexity from the training log, if present
    train_summary = {}
    tlog = os.path.join(results_dir, "train_log.json")
    if os.path.exists(tlog):
        with open(tlog) as f:
            tl = json.load(f)
        final = tl["history"][-1]
        train_summary = {
            "copy_mse": tl["copy_mse"],
            "final_val_mse": final["val_mse"],
            "beats_copy": final["val_mse"] < tl["copy_mse"],
            "final_perplexity": final["perplexity"],
        }

    report = {
        "primary_tolerance_sec": tol_p,
        "per_video": per_video,
        "micro_model_f1_primary": micro,
        "train_summary": train_summary,
    }
    with open(os.path.join(results_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"MICRO boundary F1@{tol_p}s (pooled): {micro['f1']:.3f} "
                f"(P {micro['precision']:.3f} R {micro['recall']:.3f}, {micro['tp']}/{micro['n_gt']} GT hit)")
    if train_summary:
        logger.info(
            f"Model beats copy baseline: {train_summary['beats_copy']} "
            f"(val_mse {train_summary['final_val_mse']:.5f} vs copy {train_summary['copy_mse']:.5f}), "
            f"final perplexity {train_summary['final_perplexity']:.1f}/{cfg['model']['codebook_size']}"
        )
    logger.info(f"Wrote {os.path.join(results_dir, 'eval_report.json')} and plots to {plots_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="§9 boundary-alignment + action-meaningfulness eval")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate_all(load_config(args.config))


if __name__ == "__main__":
    main()
