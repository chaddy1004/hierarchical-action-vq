# CODE_MAP — what each module does

*Reference for the `havq` package and top-level scripts. Status: **[built]** =
exists and runs; **[planned]** = designed, not yet written.*

## Pipeline at a glance

```
raw videos ──①preprocessing──►  clip features (v_t)  ──③ LAM ──►  action tokens ──► ④ BPE
   (.mp4)                        <id>.npy (n_clips,1408)          <id>.npy (ints)     (hierarchy)
                                        │
                                        └──②analysis──► diagnostics (is the signal there?)
```

- **① Preprocessing** turns each video into a sequence of frozen V-JEPA clip
  embeddings (one 1408-d vector per non-overlapping clip).
- **② Analysis** are *diagnostics* on those features (no tokenizer) — they decide
  whether/how to build the tokenizer. Summarized in `PROBE_REPORT.md`.
- **③ LAM** (latent-action model) is the LAPA-style NSVQ that turns feature pairs
  into discrete action tokens — the current PoC being built.
- **④ BPE** composition into a hierarchy — the eventual headline, not yet started.

## Directory layout

```
havq/
  data/preprocessing/   ① feature extraction
  analysis/             ② diagnostics
  model/                ③ the NSVQ latent-action model
  trainer/              ③ training + tokenization
  utils/                shared config + path helpers
run_preprocessing.py    ① entry point
make_shards.py          ① parallel-extraction sharding
```

---

## ① Preprocessing — video → clip features

- **`havq/data/preprocessing/preprocessing.py`** [built] — `Preprocessor`. Walks
  each video with a sliding window of `n_clip_frame` frames (stride `stride`),
  runs each clip through the backbone extractor, and caches one pooled vector per
  clip. **Out:** `<features_dir>/<backbone>_clip<n>_stride<s>/<id>.npy`
  `(n_clips, D)` + `<id>_meta.json` (`fps`, `n_clips`, `n_clip_frame`, …). Skips
  already-cached videos; supports a `video_ids` filter for sharded runs.
- **`havq/data/preprocessing/extractors/base_extractor.py`** [built] —
  `BaseVideoExtractor`, the abstract interface (`extract_feature(clip)->(D,)`) so
  backbones are swappable.
- **`havq/data/preprocessing/extractors/vjepa_extractor.py`** [built] — V-JEPA 2
  extractor (HF `AutoModel.get_vision_features`), **mean-pools** the patch grid to
  one vector per clip. *(This mean-pool is the lever §7 of PROBE_REPORT wants to
  replace with a learned projection.)*
- **`run_preprocessing.py`** [built] — CLI entry: `--config`, optional
  `--shard-file` (a video-id list from `make_shards.py`) and `--overwrite`.
- **`make_shards.py`** [built] — splits video ids into N duration-balanced shards
  (greedy longest-processing-time) so parallel/cluster jobs never race. **Out:**
  `shards/shard_<i>.txt`.
- Configs: `havq/data/preprocessing/configs/vjepa_clip{10,30,90}.yaml` (backbone,
  `n_clip_frame`/`stride`, paths). **clip10 is fully extracted; clip90 partial.**

Run: `python run_preprocessing.py --config .../vjepa_clip10.yaml`

---

## ② Analysis — diagnostics on the features (no tokenizer)

All read the feature cache + HD-EPIC narrations; findings in `PROBE_REPORT.md`.

- **`havq/analysis/gt.py`** [built] — maps narrations into clip-index space:
  per-clip **main-verb label** + boundary timestamps. **Out:**
  `<subdir>/<id>_gt.npz` (`labels`, `boundaries_sec`, `boundaries`).
- **`havq/analysis/probe.py`** [built] — supervised **oracle probe**: linear + MLP
  classifiers on `v_t` for verb / coarse category / participant, across three
  splits (`random_clip`, `heldout_video`, `heldout_participant`). Measures the
  *ceiling* of decodable action info and the scene confound. **Out:**
  `analysis_results/*/probe_report.json`.
- **`havq/analysis/factor.py`** [built] — projects out the 8-dim between-kitchen
  subspace and re-probes → tests whether scene is low-rank removable (it's linear
  only; nonlinear scene survives).
- **`havq/analysis/temporal_split.py`** [built] — splits `v_t` into per-video mean
  (scene) + residual (action) and re-probes → tests the "scene = slow, action =
  fast" bias. Same conclusion: linear-only removal.
- Configs: `havq/analysis/configs/{probe,probe_clip90,exp1_scale_scan}.yaml`.

Run: `python -m havq.analysis.gt --config havq/analysis/configs/probe.yaml`
(then `probe` / `factor` / `temporal_split` the same way).

---

## ③ LAM — features → discrete action tokens (the PoC)

LAPA-style: encode `(v_t, v_{t+H}) → z` (NSVQ codebook), decode `v_{t+H}` from
`v_t + z`. The decoder gets `v_t`, so the code `z` must carry the *transition*
(the action). Pairs never cross a video boundary.

- **`havq/model/vq.py`** [built] — `NSVQ`: encoder MLP `[v_t;v_{t+H}]→d`, NSVQ
  quantizer (noise-substitution, differentiable; hard nearest-code at inference),
  decoder MLP `[v_t;z]→v̂_{t+H}`, plus dead-code replacement and a `perplexity`
  helper. `l2_normalize` is applied to features so recon-MSE is comparable to the
  copy baseline. Standalone (torch/numpy only); `python -m havq.model.vq` runs a
  smoke test.
- **`havq/trainer/train_lam.py`** [planned] — pool all `(v_t, v_{t+H})` pairs from
  the clip10 cache, train NSVQ, report **val recon-MSE vs the copy baseline
  (v̂=v_t)** and **codebook perplexity** (the two health signals: must beat copy,
  perplexity ≫ 1). **Out:** `<results_dir>/nsvq.pt` + `train_log.json`.
- **`havq/trainer/tokenize_lam.py`** [planned] — apply the frozen trained model to
  every feature file → one token per `(v_t, v_{t+H})` pair. **Out:**
  `<tokens_dir>/<id>.npy` (int stream) + `<id>_tokens.json` (run-length stats).
- Config: `havq/trainer/configs/lam_clip10.yaml` (`H`, codebook size, latent dim,
  MLP widths, NSVQ schedule, train hparams). **Key setting: `H=3` (~1 s gap),** the
  fix over v1's flickery `H=1`.

Run (once built): `python -m havq.trainer.train_lam --config .../lam_clip10.yaml`
then `python -m havq.trainer.tokenize_lam --config .../lam_clip10.yaml`

---

## ④ BPE composition — tokens → hierarchy [planned]

Merge the per-video token streams bottom-up (BPE-style), then evaluate
boundary-F1 across granularity levels vs matched-random. The headline claim; not
started.

---

## Utilities & data

- **`havq/utils/config.py`** [built] — `load_config(path)` → plain dict (YAML).
- **`havq/utils/paths.py`** [built] — `feature_subdir(backbone, n_clip_frame,
  stride)` → the one canonical cache-dir name, shared by writers and readers.
- **`havq/data/dataset/`** [empty placeholder].

Data locations:
- Features: `/home/chaddy/Dataset-archive/HD-EPIC/vjepa_clip{10,90}_stride{10,90}/`
- Videos: `~/datasets/HD-EPIC/Videos/` (153 videos, ~40 h, 9 participants)
- Annotations: `~/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/`
- Archived v1 (reference; do not import): `archive/v1-derisk/havq/`

## Conventions

Copy-paste from `stepsegmenter`/archived code (no cross-repo imports); all
hyperparameters in a `config.yaml`; NSVQ (not classical VQ); `os.path` (not
`pathlib`).
