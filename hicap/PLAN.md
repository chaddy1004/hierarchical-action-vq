# hicap — PLAN

**hicap** (Hierarchical Captioning). Paper 1 of a two-paper plan; Paper 2
(in-the-wild video → robot primitives) is deferred and lives nowhere in this
package yet.

## Research goal

**Vocabulary-free, count-free hierarchical temporal action segmentation.**
Turn a long video into a *tree* of actions: atomic manipulations at the leaves,
tasks and activities at coarser levels. Two standard TAS inputs are dropped:

- **No vocabulary** — every segment is described by an open-vocabulary VLM
  caption that is *generated*, never matched to a fixed label set. Leaves are
  captioned directly; internal nodes are labeled by summarizing their children.
- **No count** — the method never consumes the ground-truth segment count N. It
  emits a full hierarchy and is read at *every* level.

**The claim that makes this publishable is the evaluation, not the pipeline.**
The pipeline (caption → agglomerative merge → summarize) is off-the-shelf parts;
a reviewer will call that engineering. The defensible core: N is an annotation
choice, so we report boundary-F1 at every hierarchy level and show the
annotator's chosen granularity is *contained* in the tree — without ever
consuming N. The multi-granularity curve is the headline, not any single number.

## What is already known (from the PoC, on HD-EPIC P09)

- The full stack runs end to end: atomic Qwen captions → agglomerative hierarchy
  → bottom-up label composition. Composition works cleanly ("Reaching into the
  dishwasher to remove a pan/plate/utensils" × 6 → "Load dishwasher contents").
- **The weak link is the grouping (boundary detection + agglomerative merge),
  not the labeling.** A node whose children are genuinely unrelated gets a vague
  parent label — the labeler faithfully reports a bad cluster, it cannot fix one.
- HD-EPIC **cannot** evaluate a hierarchy: narrations are ~1 boundary/second and
  the high-level layer is ~1/video, with no middle. A hierarchy's value is in the
  middle. So Paper 1 moves to a benchmark with real multi-granularity GT.
- Live threat carried in from the havq work: **uniform (evenly-spaced)
  segmentation beat an agglomerative hierarchy on 2 of 3 HD-EPIC videos.** Uniform
  is a real baseline, not a formality.

## Step 0 — data (blocking)

50 Salads is not on disk anywhere. Download it: overhead RGB videos (~50, ~6 min
each) + the timestamp annotations that ship **two granularities** (mid-level
~17 actions, and a coarser level). This dual-granularity GT is the whole reason
to start here.

## Experiment 1 — does the hierarchy beat the baselines on real multi-granularity GT?

**Question:** on 50 Salads, does the emitted hierarchy *contain* the annotated
granularities better than (a) uniform equal-length segmentation and (b) one
standard unsupervised-TAS method — without consuming N?

**Method:** run the ported pipeline on all 50 videos. At each hierarchy level,
extract the flat segmentation and score boundary-F1 (and the standard TAS
metrics: MoF, edit, F1@{10,25,50}) against BOTH the mid-level and coarse GT.
Report the full F1-vs-level curve. Baselines at matched segment count: uniform,
and one unsupervised method (e.g. a spectral/embedding-clustering TAS baseline).

**Decision gate:**
- The curve has a level that beats **both** uniform and the unsupervised baseline
  on ≥ the mid-level GT → the spine holds; scale to a second, diverse benchmark
  (Breakfast or an ego set) for the submission.
- No level beats uniform → the grouping is the problem (as the PoC suggested).
  Fix boundary detection / the merge criterion before anything else; do not add
  datasets or write anything up.

**VERDICT (2026-07-20): PASS.** Ward hierarchy on standard I3D features beats
uniform AND random at every level from 2–176 segments, peak gain +0.101 (mid) /
+0.118 (verb) at 19 segments, on all 50 videos. Full result + ablation in
`RESULT_gate.md`. Caveat: absolute F1 peaks at ~0.24 — the lever is now the
feature/boundary signal, not the eval. (Only uniform/random baselines run so far;
a published unsupervised-TAS baseline still owed.) Next fork listed in RESULT_gate.md.

**Cost:** captioning ~50×6 min of video is the expensive part (one VLM pass);
everything downstream is cheap CPU.

## Explicitly NOT now

Robotics / robot-primitive transfer (Paper 2). LAPA action latents (shelved;
only revisit if grouping needs a better boundary signal, or for Paper 2). The
GNN refinement, multi-encoder "single feature lineage", and eigengap count
selection from stepsegmenter — port the working spine only, leave the half-built
machinery in that repo. A second benchmark, ablations, and the final metric
table — all gated on Experiment 1's verdict.
