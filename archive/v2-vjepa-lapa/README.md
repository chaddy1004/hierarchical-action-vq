# archive/v2-vjepa-lapa

Snapshot of the repo before the 2026-07-20 reset. Archived whole, structure intact,
so any single file can be pulled back with a plain copy:

    cp archive/v2-vjepa-lapa/havq/analysis/hierarchy.py havq/analysis/

Full working state is also on branch `use-lapa` (commit `b8e3521`) if something is
missing here.

Two separate lines of work are in this directory. Neither reached a positive result.

---

## Line A — V-JEPA features + a from-scratch NSVQ tokenizer

The original plan (`PLAN.md`): frozen V-JEPA clip features, train a small NSVQ to
discretize them into action tokens, then BPE-merge the token stream into a
hierarchy.

Built and run: feature extraction (`havq/data/preprocessing/`), the supervised
probes (`havq/analysis/{probe,factor,temporal_split}.py`), and the NSVQ model
(`havq/model/vq.py`). The trainer (`havq/trainer/`) was designed but never run.

Findings are in **`PROBE_REPORT.md`** — the short version is that frozen `v_t` is
scene-dominated, coarse-action signal survives a linear probe but fine verbs do
not, and scene is not linearly removable. `CODE_MAP.md` maps the module layout.

## Line B — LAPA latent actions + bottom-up hierarchy

Replaced the feature-extraction and tokenizer-training stages with LAPA's
pretrained LAQ inverse-dynamics model, then built the hierarchy by agglomerative
merging instead of BPE. Motivation: LAPA is a foundation model that already learned
an action representation, so a **single video can be segmented with no corpus** —
removing the usual requirement for a set of similar videos to learn a vocabulary
from. That independence claim was the intended contribution.

Files: `havq/model/laq/` (vendored LAPA, see below),
`havq/data/preprocessing/extractors/lapa_extractor.py`,
`havq/data/preprocessing/lapa_tokenize.py`, `havq/analysis/hierarchy.py`,
`run_lapa_tokenize.py`, `run_hierarchy.py`.

### What was established (all verified, on 3 annotated HD-EPIC videos)

**The model is small and cheap.** LAQ is 343.8 M params, *not* the 7B VLM — the
7B is a separate JAX model for robot control and is never needed here. Checkpoint
`laq_openx.pt` is 1.38 GB (= 343.8 M x 4 bytes exactly), peaks at 2.06 GB VRAM at
batch 64, and runs ~8x realtime, bottlenecked on video decoding rather than GPU.
The full 153-video / 40 h corpus is ~5 GPU-hours.

**A large constant offset must be removed first.** `||mean||` is ~90% of a typical
row norm, and per-video means are mutually cosine **0.991-0.996** — the same vector
in every kitchen, so it is a fixed artifact of LAQ's encoder, not scene content.
Uncentered, every delta looks similar to every other one and all cosine statistics
are meaningless. Centering per-video and globally give nearly identical results, so
centering costs nothing in corpus-independence.

**Centered deltas carry real temporal structure.** Lag-1 cosine minus a
time-shuffled control (which preserves the marginal distribution and destroys only
time order):

| tokens | P05 | P06 | P09 |
|---|---|---|---|
| 1 s   | +0.271 | +0.157 | +0.194 |
| 0.5 s | +0.322 | +0.321 | +0.219 |

Finer tokens are *more* coherent.

**No scene confound.** Within-video cosine (+0.023/+0.007/+0.008) is
indistinguishable from across-video (-0.010/-0.009/-0.004). The deltas do not
cluster by kitchen — a difference of two frames in one scene cancels the static
content. This is the property that makes LAPA's space usable across videos and
contrasts sharply with Line A's scene-dominated V-JEPA features.

**Low-dimensional.** 20 of 128 dims carry 90% of variance.

**LAPA's own codebook is too fine to use directly.** 132/161, 253/433, 120/161
distinct code-tuples per video — nearly every second gets a unique word out of the
8^4 = 4096 space, so nothing recurs and BPE has nothing to merge. Using the native
per-cell vocabulary instead (4 cells x 8 entries) fixes recurrence (top bigram
recurs 14-78x) but the streams barely hold still: run length 1.35 vs 1.22 shuffled.
LAPA's 12-bit quantization discards most of the temporal structure its own
continuous representation contains.

**Latent actions compose.** LAPA-on-endpoints vs the sum of the fine tokens
spanning the same stretch, cosine over a shuffled control:

| span | P05 | P06 | P09 |
|---|---|---|---|
| 1 s | +0.83 | +0.74 | +0.84 |
| 2 s | +0.60 | +0.47 | +0.64 |
| 4 s | +0.40 | +0.22 | +0.44 |
| 8 s | +0.21 | +0.16 | +0.29 |

Strong at short spans, decaying by 8 s. So pooling fine tokens is a sound
representation for a merged segment near the bottom of a tree, and drifts near the
top; re-running LAPA at wide gaps is worse, being far outside its training regime.

### Why it was stopped

The hierarchy does not beat **evenly spaced boundaries**. Against a uniform grid
(no model at all), best gain per level:

| video | vs uniform |
|---|---|
| P05 | loses at 5 of 7 levels, to -0.094 |
| P06 | ties (+/-0.03) |
| P09 | **+0.188 / +0.197** at 13-20 segments, loses at fine levels |

Only P09 wins, and only at the coarse end (8-12 s segments).

A large part of this is an evaluation mismatch, not necessarily a method failure.
HD-EPIC narration boundaries have a **median gap of 0.95-1.36 s** — nearly as fine
as the tokens, and near-periodic, which makes a uniform grid close to optimal by
construction and leaves almost no headroom (random placement alone reaches F1
0.5-0.6). The coarser `high-level/activities/` layer gives only **1-2 segments per
video**, i.e. zero or one internal boundary. HD-EPIC has ~1 boundary/second or
~1 boundary/video and nothing between, and a hierarchy's value is entirely in that
middle.

The recommendation at the time of archiving: evaluate on a dataset with genuine
multi-granularity ground truth (50 Salads has explicit high/mid/low levels;
Breakfast Actions similar). HD-EPIC remains fine for the single-video/no-corpus
claim but cannot score a hierarchy.

### Never tested

- BPE / merge-by-recurrence vs. Ward / merge-by-similarity. Only the latter ran;
  `havq/analysis/bpe_eval.py` (written for Line A) was never run against LAPA tokens.
- Vocabulary size per level — fixed at 12, never swept.
- Anything beyond 3 videos. One video showing an effect out of three is thin.
- Pair gaps other than 0.5 s and 1 s.

### Note on the vendored LAPA code

`havq/model/laq/` holds 3 files copied from LatentActionPretraining/LAPA at commit
`6a2dfb9877f8f5d45acd8ea4f91cb502f1d9c9b3`, byte-identical to upstream except two
import lines. Upstream's own `laq_model/__init__.py` was deliberately dropped: it
imports the trainer, which drags in accelerate / ema-pytorch / wandb / tensorflow.
Keeping only the inference files cuts the dependency chain to torch + einops +
beartype and avoids the torch-plus-tensorflow segfault seen on GPU nodes.

Checkpoint is not in the repo: `/home/chaddy/weights/laq_openx.pt`, from
`https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/laq_openx.pt`.

Cached outputs (not in the repo) are at
`/home/chaddy/Dataset-archive/HD-EPIC/lapa_clip{15,30}_stride{15,30}/`.

---

## Also archived here

Compute Canada / SLURM tooling (`cc_*.sh`, `*.slurm`, `run_protected.sh`,
`JOBS/`, `shards/`, `make_shards.py`, `UV_ON_COMPUTE_CANADA.md`) and the research
notes (`RESEARCH.md`, `RELATED_WORKS.md`, `RESEARCH_STRUGGLE.md`). None of it is
specific to either line above and it is all reusable as-is.

`PLAN.md` is the pre-reset plan and is now stale — it describes Line A's two
experiments.
