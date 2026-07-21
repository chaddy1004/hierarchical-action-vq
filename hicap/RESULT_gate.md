# Experiment 1 gate — RESULT (2026-07-20)

**PASS.** On 50 Salads, the contiguity-constrained agglomerative hierarchy
contains granularity levels whose boundaries beat both uniform segmentation and a
matched-count random baseline — without ever consuming the ground-truth segment
count. This clears the make-or-break gate in `PLAN.md`.

## Setup

- Data: 50 Salads, canonical MS-TCN release (Zenodo 3625992). 50 videos, I3D
  features `(2048, T)`, ~6.5 min each, 30 fps. Using the standard precomputed
  features makes the comparison apples-to-apples with unsupervised-TAS baselines
  and needs no VLM (the expensive step stays deferred until warranted).
- Method: Ward-linkage agglomerative merging of the per-frame feature sequence,
  contiguity-constrained (only time-adjacent segments merge). Every horizontal cut
  of the dendrogram is one granularity level.
- Two GT granularities: `groundTruth` (mid, ~17 action classes) and a derived
  `verb` level (mid labels coarsened to their verb prefix: cut_tomato -> cut).
- Metric: boundary-detection F1, tolerance 1.0 s (30 frames). Baselines placed
  with the SAME boundary count as the model at each level: **uniform** (equal
  spacing) and **random** (uniform-random, 200 trials).

## The number that matters

Mid-level GT, mean over 50 videos, F1 vs level (segment count):

| segs |  model | uniform | random | gain vs uniform |
|-----:|-------:|--------:|-------:|----------------:|
|   4  | 0.076  | 0.042   | 0.028  | +0.034 |
|   9  | 0.151  | 0.082   | 0.061  | +0.069 |
|  13  | 0.182  | 0.089   | 0.077  | +0.093 |
|  19  | 0.215  | 0.114   | 0.095  | **+0.101** |
|  40  | 0.235  | 0.136   | 0.124  | +0.099 |
|  84  | 0.234  | 0.161   | 0.136  | +0.073 |
| 121  | 0.214  | 0.181   | 0.130  | +0.033 |
| 176  | 0.178  | 0.175   | 0.119  | +0.003 |
| 256  | 0.136  | 0.138   | 0.103  | -0.002 |

The `verb` level is the same shape (peak gain +0.118 at 19 segs).

**Two things make this a real result, not a cherry-pick:**
1. The model beats uniform at *every* level from 2 to ~176 segments, not one
   tuned point. The gain peaks in the middle (9-40 segments), exactly where the
   annotated granularity lives (50 Salads averages ~15-20 actions/video), and
   decays to a tie only at absurdly fine levels (256 segs in a video with ~15
   real boundaries — pure noise, uniform ties by construction).
2. 50 Salads boundaries are ~28 s apart (not HD-EPIC's ~1/s), so uniform is NOT
   trivially optimal here — there is genuine headroom, and the hierarchy uses it.

## Ablation (peak over levels, mean of 50 videos)

| linkage        | peak model F1 | peak gain vs uniform |
|----------------|--------------:|---------------------:|
| **ward**       | 0.238         | **+0.101**           |
| ward + L2-norm | 0.240         | +0.092               |
| average        | 0.207         | +0.070               |
| average + L2   | 0.223         | +0.084               |

Plain Ward (no normalization) is best or tied-best on both metrics. The config is
validated; L2 and group-average linkage do not help.

## Honest caveats

- **Absolute F1 is low (~0.24 peak).** The claim here is only "beats the
  baselines," which holds clearly (~2x uniform at the peak). But 0.24 is far from
  a strong segmentation. That ceiling is raw I3D + plain Ward with no learning and
  no boundary model — it is the naive floor of the approach, not its limit.
- "Best level" selection is oracle-ish, but the point is exactly that the
  hierarchy *contains* a good level; the full curve (above) is the honest report,
  and it dominates uniform across the whole meaningful range.
- 50 Salads is a single recipe. This validates the machinery and the count-free
  eval; it does not show task diversity. A second, diverse benchmark is needed
  for the submission.

## Next fork (per PLAN gate: PASS -> scale)

The spine holds, so the direction is alive. The low absolute F1 says the lever is
the **feature / boundary signal**, not the eval. Candidate next steps, in
rough priority:
1. Better features for the same hierarchy: V-JEPA (have the extractor) or the
   method's own VLM-caption embeddings, replacing raw I3D.
2. A real boundary/change-point detector feeding the hierarchy, instead of pure
   bottom-up agglomeration from frame 0.
3. A second benchmark (Breakfast, or an ego set that also seeds Paper 2) to show
   generality.
4. Standard TAS metrics (MoF, edit, F1@{10,25,50}) alongside boundary-F1, and a
   published unsupervised-TAS baseline (not just uniform/random) for comparison.

Reproduce: `python run_gate.py --config hicap/configs/gate_50salads.yaml`
Report: `hicap/results/gate_report.json`
