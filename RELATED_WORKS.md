# RELATED_WORKS.md

Running log of related work: what exists, how close it is, and how we
differentiate. New sweeps get a dated section; the reading *checklist* (what
still must be reviewed) lives in `RESEARCH.md`.

---

## Sweep 2026-07-05 (quick web pass — searches + abstract reads, not full reads)

**Verdict: the exact combination is still open, but three recent papers each
hold one piece. These are must-cite, must-differentiate:**

1. **"From Observation to Action: Latent Action-based Primitive Segmentation
   for VLA Pre-training in Industrial Settings"** (arXiv 2511.21428, Nov
   2025) — the closest neighbor on the tokenization side. Latent-action motion
   tokenizer + "Latent Action Energy" with a hysteresis controller →
   unsupervised segmentation of industrial video into action primitives, for
   VLA pretraining data. **No hierarchy, no composition, no TAS-benchmark
   eval** (VLM-judged semantic coherence instead). Our differentiators:
   multi-granularity merge tree, count-free evaluation on TAS benchmarks.
2. **"Unsupervised Skeleton-Based Action Segmentation via Hierarchical
   Spatiotemporal Vector Quantization"** (arXiv 2604.15196) — hierarchical VQ
   for unsupervised action segmentation, but **skeleton input** and a *fixed
   two-level* quantizer (subaction → action), not open-ended composition of a
   token stream. Also relevant: **"Skeleton Motion Words"** (arXiv 2508.04513)
   — the "motion words" metaphor is already in use for skeleton TAS.
3. **"Unsupervised Hierarchical Skill Discovery"** (arXiv 2601.23156, Jan
   2026) — grammar induction (modified **Sequitur** — BPE's close cousin:
   repeated digrams → non-terminal rules) over skills discovered from
   pixel-based trajectories. **But**: game/RL domains (Craftax, Minecraft),
   evaluated on downstream RL, not video TAS. Proof the "compress symbol
   streams into a hierarchy" move is in the air; Sequitur is also a candidate
   alternative to vanilla BPE for our composition step.

**Other confirmations from the sweep:**

- **BPE beyond text is established machinery**: BPE motif discovery on
  discretized time series (arXiv 2505.14411), multidimensional BPE for visual
  tokens (arXiv 2411.10281). Cite as analogy; none do video segmentation.
- **Speech-unit → word discovery is a mature field, as suspected**:
  unsupervised word segmentation from discrete units (arXiv 2106.04298;
  arXiv 2202.11929, DP + self-supervised scoring), ZeroSpeech 2020 (arXiv
  2010.05967). Frame our work as the video analog and mine it for
  composition algorithms beyond BPE.
- **Backbone good news for H1**: V-JEPA 2 (arXiv 2506.09985) probes strongly
  on *motion* tasks (SSv2 75.3%, Diving-48, Jester) vs appearance tasks —
  the features do encode motion, not just scene. Direct comparison paper:
  "Temporal vs. Spatial: DINOv3 vs V-JEPA2" (arXiv 2509.21595) — read before
  any backbone swap.
- **Recent unsupervised TAS baselines to add to group B**: CLOT (Closed-Loop
  Optimal Transport, arXiv 2507.03539), plus classic CTE / TW-FINCH lines.
- **Pre-deep prior art to cite**: "Learning action symbols for hierarchical
  grammar induction" (IEEE, 2013) — quantized motion symbols + grammar
  induction; the idea's ancestry, not a competitor.

**Implication:** the surviving claim is specifically *frozen
video-foundation-model latent-action tokens + full merge-tree composition +
count-free multi-granularity evaluation on standard TAS benchmarks*. Three
groups are one step away — velocity matters.

---

## Differentiation strategy (2026-07-05)

Read the neighbors coldly, by the **question each answers** — not the
mechanism:

| Work | Question it answers | What it lacks |
|---|---|---|
| Industrial latent-action segmentation (2511.21428) | "Can I harvest VLA pretraining clips from unlabeled video?" | Flat segments only; no hierarchy; no TAS benchmarks (VLM-judged eval) |
| Hierarchical spatiotemporal VQ (2604.15196) | "Does a two-level quantizer segment skeleton sequences?" | Hierarchy is *architectural* (fixed 2 levels), not emergent; skeleton input, not RGB |
| Unsupervised hierarchical skill discovery (2601.23156) | "Does grammar structure accelerate downstream RL?" | Game domains; no video TAS evaluation; hierarchy serves control, not segmentation |

**Our question — untouched by all three:** *does unsupervised compositional
structure align with the granularity choices human annotators make?* —
evaluated count-free and vocabulary-free, at **every** level of an emergent
hierarchy, on standard TAS benchmarks. Differentiate on the claim and the
question; mechanisms (VQ, BPE) get reinvented constantly, claims are what a
paper owns.

**How to use each neighbor:**

1. **Load-bearing citation, not rival**: 2511.21428 is *independent evidence
   for our H1* — cite as "latent-action tokens support flat primitive
   segmentation; we show they *compose*."
2. **Baseline row**: if Latent Action Energy (2511.21428) is reproducible,
   run it on our benchmarks as the flat-segmentation baseline. The scariest
   neighbor becomes a table row.
3. **Imported ablations**: Sequitur (2601.23156) vs BPE vs similarity-linkage
   as the composition ablation; speech-unit segmentation algorithms
   (2106.04298, 2202.11929) as further candidates.
4. **Own the evaluation depth they won't build**: boundary-F1 across all
   hierarchy levels with matched-budget baselines on the TAS standards.
   VLA/RL-motivated groups don't invest in TAS eval infrastructure; that
   depth is the moat.

**Differentiation sentences (keep sharp; these become the intro):**

- *Unlike 2511.21428, which extracts flat action primitives to feed VLA
  pretraining and evaluates via VLM-judged coherence, we compose latent-action
  tokens into a full granularity hierarchy and evaluate boundary alignment
  against human annotations at every level, count-free.*
- *Unlike 2604.15196, whose two granularity levels are fixed by architecture
  and whose input is skeletons, our hierarchy is emergent — every merge is a
  level — and operates on RGB video via a frozen foundation model.*
- *Unlike 2601.23156, which induces skill grammars in game environments to
  accelerate RL, we target temporal action segmentation of real-world video
  and measure alignment with human annotation granularity.*
