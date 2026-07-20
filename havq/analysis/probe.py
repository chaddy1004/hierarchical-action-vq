"""
havq/analysis/probe.py
======================
Oracle probe (diagnostics, NOT the method). Measures how much ACTION (verb)
information is decodable from a single frozen V-JEPA clip embedding v_t -- the
ceiling any unsupervised tokenizer is aiming at -- and how much v_t is instead
dominated by scene/kitchen identity.

For each split we train two probes on v_t -> verb (106 classes):
    linear : logistic regression (tests linear decodability)
    mlp    : one hidden layer (tests nonlinear decodability -- a higher ceiling)
reported as top-1 / top-5 accuracy and macro-F1, against a majority-class
baseline and uniform chance.

Three splits reveal how scene-dependent the verb signal is:
    random_clip         : clips split ignoring video -> scene leaks freely (optimistic)
    heldout_video       : whole videos held out, stratified by participant
    heldout_participant : whole kitchens (participants) never seen in training (strictest)
A big drop random_clip -> heldout_participant means the "verb signal" is largely
scene memorization. Separately, a v_t -> participant probe measures directly how
strongly the feature encodes the kitchen (the scene confound).

Reads cached features (<subdir>/<video_id>.npy) and per-clip labels
(<subdir>/<video_id>_gt.npz from havq/analysis/gt.py). No VQ, no tokenizer.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from havq.utils.config import load_config
from havq.utils.paths import feature_subdir

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ data

def load_verb_categories(csv_path: str):
    """Map each verb-class id to a coarse category id (HD_EPIC_verb_classes.csv
    `category` column). Returns (id2catid int array indexed by verb id, cat_names)."""
    df = pd.read_csv(csv_path)
    id2cat_name = dict(zip(df["id"].astype(int), df["category"].astype(str)))
    cat_names = sorted(set(id2cat_name.values()))
    cat2id = {c: i for i, c in enumerate(cat_names)}
    id2catid = np.full(max(id2cat_name) + 1, -1, dtype=np.int64)
    for vid_, cname in id2cat_name.items():
        id2catid[vid_] = cat2id[cname]
    return id2catid, cat_names


def load_dataset(cfg: dict):
    """Load every labeled (non-background) clip across the feature cache.

    Returns X (N, D) float32 cpu tensor; verb, cat, part, vid (N,) int64 tensors;
    a participant name->id dict; and the coarse category names.
    """
    features_root = cfg["paths"]["features_root"]
    subdir = feature_subdir(cfg["model"], cfg["scale"]["n_clip_frame"], cfg["scale"]["stride"])
    fdir = os.path.join(features_root, subdir)

    gt_files = sorted(glob.glob(os.path.join(fdir, "*_gt.npz")))
    if not gt_files:
        raise FileNotFoundError(f"No *_gt.npz in {fdir} -- run havq/analysis/gt.py first")

    X, verb, part, vid = [], [], [], []
    participants: dict[str, int] = {}
    for gi, gt_path in enumerate(gt_files):
        video_id = os.path.basename(gt_path)[: -len("_gt.npz")]
        feat_path = os.path.join(fdir, video_id + ".npy")
        if not os.path.exists(feat_path):
            continue
        feats = np.load(feat_path).astype(np.float32)
        labels = np.load(gt_path)["labels"]
        n = min(len(feats), len(labels))
        feats, labels = feats[:n], labels[:n]
        keep = labels >= 0  # drop background clips
        if not keep.any():
            continue
        pname = video_id.split("-")[0]
        pid = participants.setdefault(pname, len(participants))
        X.append(feats[keep])
        verb.append(labels[keep].astype(np.int64))
        part.append(np.full(int(keep.sum()), pid, dtype=np.int64))
        vid.append(np.full(int(keep.sum()), gi, dtype=np.int64))

    X = torch.from_numpy(np.concatenate(X))
    verb = torch.from_numpy(np.concatenate(verb))
    part = torch.from_numpy(np.concatenate(part))
    vid = torch.from_numpy(np.concatenate(vid))

    id2catid, cat_names = load_verb_categories(cfg["paths"]["verb_classes_csv"])
    cat = torch.from_numpy(id2catid[verb.numpy()])

    logger.info(f"Loaded {len(X)} labeled clips, D={X.shape[1]}, "
                f"{len(participants)} participants, {int(vid.max()) + 1} videos, "
                f"{int(verb.max()) + 1} verbs, {len(cat_names)} categories")
    return X, verb, cat, part, vid, participants, cat_names


def make_splits(cfg: dict, part, vid, participants, rng):
    """Return {split_name: (train_idx, test_idx)} as int64 numpy arrays."""
    frac = cfg["probe"]["heldout_video_fraction"]
    part_np, vid_np = part.numpy(), vid.numpy()
    n = len(part_np)
    splits = {}

    # random clip: ignores video boundaries -> scene leaks freely
    perm = rng.permutation(n)
    ntest = int(frac * n)
    splits["random_clip"] = (perm[ntest:], perm[:ntest])

    # heldout video: hold out a fraction of videos per participant (stratified)
    vids = np.unique(vid_np)
    vid2part = {int(v): int(part_np[vid_np == v][0]) for v in vids}
    test_vids = []
    for p in np.unique(part_np):
        pv = np.array([v for v in vids if vid2part[v] == p])
        rng.shuffle(pv)
        k = max(1, int(frac * len(pv)))
        test_vids.extend(pv[:k].tolist())
    tmask = np.isin(vid_np, test_vids)
    splits["heldout_video"] = (np.where(~tmask)[0], np.where(tmask)[0])

    # heldout participant: whole kitchens never seen in training
    hp_ids = [participants[nm] for nm in cfg["probe"]["heldout_participants"] if nm in participants]
    pmask = np.isin(part_np, hp_ids)
    splits["heldout_participant"] = (np.where(~pmask)[0], np.where(pmask)[0])

    return splits


# ---------------------------------------------------------------- probe

class Probe(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int | None):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))
        else:
            self.net = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.net(x)


def _metrics(logits: torch.Tensor, y_true: torch.Tensor, topk) -> dict:
    y = y_true.numpy()
    top = logits.topk(max(topk), dim=1).indices.numpy()
    out = {}
    for k in topk:
        out[f"top{k}"] = float(np.mean([y[i] in top[i, :k] for i in range(len(y))]))
    out["macro_f1"] = float(f1_score(y, top[:, 0], average="macro", zero_division=0))
    return out


def train_probe(X, y, tr_idx, te_idx, n_classes, hidden, cfg, device) -> dict:
    pcfg = cfg["probe"]
    Xtr, Xte = X[tr_idx].clone(), X[te_idx].clone()
    ytr, yte = y[tr_idx], y[te_idx]

    if pcfg["standardize"]:
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte = Xte.to(device)

    model = Probe(X.shape[1], n_classes, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pcfg["lr"], weight_decay=pcfg["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    bs, ntr = pcfg["batch_size"], len(Xtr)

    model.train()
    for _ in range(pcfg["epochs"]):
        perm = torch.randperm(ntr, device=device)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(Xte[i:i + bs]) for i in range(0, len(Xte), bs)]).cpu()

    m = _metrics(logits, yte, pcfg["topk"])
    # majority-class baseline (predict train mode) and uniform chance
    majority = int(torch.bincount(ytr.cpu()).argmax())
    m["majority_top1"] = float((yte == majority).float().mean())
    m["chance_top1"] = 1.0 / n_classes
    m["n_train"], m["n_test"], m["n_classes"] = len(tr_idx), len(te_idx), n_classes
    return m


def pick_device(pref: str) -> str:
    if pref == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 4 * 1024 ** 3:
            return "cuda"
        logger.warning(f"cuda has only {free / 1e9:.1f} GB free; falling back to cpu")
    return "cpu"


# ---------------------------------------------------------------- driver

def run(cfg: dict) -> dict:
    torch.manual_seed(cfg["probe"]["seed"])
    rng = np.random.default_rng(cfg["probe"]["seed"])
    device = pick_device(cfg["probe"]["device"])
    logger.info(f"device: {device}")

    X, verb, cat, part, vid, participants, cat_names = load_dataset(cfg)
    n_verbs = int(verb.max()) + 1
    n_cats = len(cat_names)
    n_parts = len(participants)
    splits = make_splits(cfg, part, vid, participants, rng)

    results = {"n_verbs": n_verbs, "n_categories": n_cats, "n_participants": n_parts,
               "verb": {}, "category": {}, "participant": {}}

    # action probes (fine verb + coarse category) across all three splits, linear + mlp
    for tgt_name, y, n_classes in (("verb", verb, n_verbs), ("category", cat, n_cats)):
        for split_name, (tr, te) in splits.items():
            if len(te) == 0:
                logger.warning(f"{tgt_name} [{split_name}]: empty test split, skipping")
                continue
            results[tgt_name][split_name] = {}
            for kind, hidden in (("linear", None), ("mlp", cfg["probe"]["mlp_hidden"])):
                m = train_probe(X, y, tr, te, n_classes, hidden, cfg, device)
                results[tgt_name][split_name][kind] = m
                logger.info(
                    f"{tgt_name.upper():>8} [{split_name:>18}] {kind:<6} "
                    f"top1 {m['top1']:.3f} top5 {m['top5']:.3f} macroF1 {m['macro_f1']:.3f} "
                    f"(majority {m['majority_top1']:.3f}, chance {m['chance_top1']:.4f})"
                )

    # scene confound: how decodable is the kitchen (participant) from v_t?
    # run on random_clip (in-distribution upper bound of scene encoding).
    tr, te = splits["random_clip"]
    for kind, hidden in (("linear", None), ("mlp", cfg["probe"]["mlp_hidden"])):
        m = train_probe(X, part, tr, te, n_parts, hidden, cfg, device)
        results["participant"][kind] = m
        logger.info(
            f"PART [       random_clip] {kind:<6} "
            f"top1 {m['top1']:.3f} macroF1 {m['macro_f1']:.3f} "
            f"(majority {m['majority_top1']:.3f}, chance {m['chance_top1']:.4f})"
        )

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "probe_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Oracle probe: verb ceiling + scene confound from V-JEPA v_t")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
