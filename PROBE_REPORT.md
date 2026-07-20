# V-JEPA Action-Token Feasibility Probe — Report & Session Context

*Experiment run 2026-07-13. Standalone report (printable) that also serves as
onboarding context for a fresh LLM session.*

**Version: v3 (2026-07-13).** Changelog at the bottom.

---

## 0. Orientation (read first if you're new to this project)

**Goal of the project (`hierarchical-action-vq`, package `havq`).** Vocabulary-free,
count-free **temporal action segmentation with a discovered hierarchy**. The
intended pipeline:

1. **Tokenize** — turn a video into a stream of discrete "action tokens" using a
   small VQ model on top of a **frozen** V-JEPA 2 video encoder (no pixel
   generation; everything happens in V-JEPA feature space).
2. **Compose** — merge that token stream bottom-up, BPE-style, so every cut of
   the resulting merge tree is a segmentation at one granularity. The tree *is*
   the multi-granularity output.

The publishable claim is the **composition into a hierarchy**, not the tokenizer.
The tokenizer is enabling infrastructure.

**Why V-JEPA (not pixel-space LAPA-style).** We have limited data (~40 h HD-EPIC,
egocentric kitchen video) but decent compute. A frozen internet-scale encoder
buys data efficiency and (hopefully) some robustness to egocentric camera
motion, at the cost of being at the mercy of what its features encode.

**Key terms.**
- `v_t` = one **mean-pooled** V-JEPA embedding of a short clip (a single 1408-d
  vector, *not* the pixel clip). "clipN" = N-frame clips at 30 fps (clip10 = ⅓ s,
  clip30 = 1 s, clip90 = 3 s), non-overlapping.
- **LAM (latent action model)** = LAPA-style inverse+forward dynamics with a VQ
  bottleneck: infer a discrete code from `(v_t, v_{t+H})`, reconstruct `v_{t+H}`
  from `v_t + code`; the code is forced to carry the *change*.
- Two tokenization options debated: **transition tokens** (quantize the change
  between clips — what v1 did) vs **clip tokens** (quantize `v_t` directly). v1's
  transition tokens flickered every step; the working hypothesis is that a
  per-clip "what action is happening" token is the better unit for BPE.

**State of the world.**
- v1 (archived in `archive/v1-derisk/`) built the full LAM+eval pipeline. Result:
  at ⅓ s clips with H=1 the transition tokens flicker every step and boundary F1
  ties random — but tokens carry *some* clip-level action content (verb NMI
  0.27–0.48).
- Features are extracted and cached at
  `/home/chaddy/Dataset-archive/HD-EPIC/vjepa_clip10_stride10/` (all 153 videos)
  and `.../vjepa_clip90_stride90/` (partial, ~97 videos / participants P01–P05).
- **This experiment** is the first step of the current phase: before building any
  tokenizer, measure whether the action signal is even *present and recoverable*
  in `v_t`.

**Data & code map.**
- Features: `<root>/vjepa_clip{10,90}_stride{10,90}/<id>.npy` `(n_clips, 1408)` +
  `<id>_meta.json`. Root = `/home/chaddy/Dataset-archive/HD-EPIC`.
- Labels: `HD_EPIC_Narrations.pkl` (59 k narrations), `HD_EPIC_verb_classes.csv`
  (106 verbs, each with a coarse `category`). Both under
  `~/datasets/HD-EPIC/hd-epic-annotations/narrations-and-action-segments/`.
- Analysis code: `havq/analysis/{gt,probe,factor}.py`, configs in
  `havq/analysis/configs/`. Raw outputs in `analysis_results/*/`.

---

## 1. Assumptions

1. V-JEPA `v_t` should contain *some* action-relevant information (it is
   motion-trained, and v1 already found weak clip-level verb content).
2. A **supervised probe** on `v_t` is a valid **ceiling / kill-test**: it
   upper-bounds what *any* unsupervised tokenizer could extract. If a supervised
   probe cannot decode actions, the direction is dead; if it can, any failure of
   an unsupervised tokenizer is an *extraction* problem, not a *presence* problem.
   (A naive-VQ failure, by contrast, only kills one implementation — it is not a
   valid kill-test.)
3. **Participant/kitchen identity** (`P0x` prefix) is an acceptable proxy for
   "scene."
4. The **main verb** of the narration covering a clip's center is an acceptable
   per-clip action label; background clips (no narration) are excluded.

## 2. Hypothesis

> A single frozen V-JEPA clip embedding `v_t` carries enough action information —
> recoverable and **generalizable across kitchens** — to be the basis for
> discrete action tokens.

Sub-questions: *How much* signal? *Fine* (verb) vs *coarse* (category)? *How
scene-entangled* is it? Does *clip length* change it? Is the scene *removable*?

## 3. Method (concise)

- **Dataset:** map narrations into clip-index space (clip *j* center = `(j+0.5) *
  n_clip_frame / fps`; assign the covering narration's main verb; later-starting
  narration wins on overlap). clip10 → **341,167 labeled clips**, 153 videos, 9
  kitchens, 105 verbs present, 13 coarse categories. *Only the verb is an
  annotation*; **category** is a deterministic coarsening of the verb (the CSV's
  `category` column), and **participant/kitchen** is metadata (video-id prefix),
  not an annotation. A narration spans ~1–3 s, so several consecutive ⅓ s clips
  share one verb.
- **Probe:** z-score `v_t` (train-split stats), then train two classifiers —
  **linear** (logistic regression) and **MLP** (one 512-d hidden layer) — for each
  target (verb / category / participant). Report top-1, top-5, **macro-F1** (equal
  weight per class, so not dominated by frequent verbs), vs a majority-class
  baseline and uniform chance.
- **Three train/test splits** (the core instrument):
  - `random_clip` — clips shuffled 80/20; train & test can share a video → **scene
    can leak** (optimistic).
  - `heldout_video` — whole videos held out; same kitchens still in train.
  - `heldout_participant` — whole kitchens (P06, P07) unseen → **honest
    cross-scene** number.
  - A high `random_clip` score that **collapses** on `heldout_participant` means
    the probe was reading *scenes*, not *actions*. That collapse is the measure of
    scene-entanglement.
- **Scale check:** rerun the whole pipeline on the clip90 (3 s) partial cache.
- **Scene factoring (`factor.py`):** compute the 8-dim between-kitchen mean
  subspace, **project it out** of every `v_t`, and re-probe — does linear kitchen
  decoding die while action decoding survives?

## 4. Observations

**clip10 (⅓ s), headline numbers:**

| target | split | linear top1 | mlp top1 | mlp macroF1 | majority top1 |
|---|---|---|---|---|---|
| verb (105) | random_clip | 0.516 | **0.668** | 0.647 | 0.157 |
| verb (105) | heldout_video | 0.353 | 0.330 | 0.131 | 0.175 |
| verb (105) | **heldout_participant** | 0.245 | 0.224 | **0.047** | 0.203 |
| category (13) | random_clip | 0.520 | **0.686** | 0.673 | 0.204 |
| category (13) | heldout_video | 0.410 | 0.402 | 0.323 | 0.241 |
| category (13) | **heldout_participant** | 0.365 | 0.287 | **0.208** | 0.246 |
| participant (9) | random_clip | **0.999** | 1.000 | — | 0.178 |

- **Scene is near-perfectly encoded** (kitchen top1 ≈ 1.0).
- **Verb decoding collapses across kitchens** (macroF1 0.65 → 0.05); **coarse
  category survives** (macroF1 → 0.21–0.25; top1 0.37 vs 0.25 majority).
- **MLP beats linear only in-distribution** (random_clip), never on the held-out
  splits → the extra capacity was memorizing scene, not learning transferable
  actions.

**Scale (clip90, 3 s; 21 k clips, 5 kitchens):** no material gain at the
generalizing splits (verb heldout_video 0.40 vs clip10 0.33; category
heldout_video 0.45 vs 0.40). Kitchen still ≈ 0.99. *Caveat: 16× fewer clips and
5 vs 9 kitchens — suggestive, not airtight.*

**Scene factoring (remove 8-dim between-kitchen subspace):**

| | linear kitchen | mlp kitchen | category heldout_part (macroF1) |
|---|---|---|---|
| raw | 0.999 | 1.000 | 0.249 / 0.214 |
| scene-removed | **0.174** (≈ chance 0.111) | **1.000** (unchanged) | 0.258 / 0.234 (unchanged) |

- Linear scene info is **low-rank (8 dims) and removable without hurting action**.
- But the **MLP still decodes the kitchen perfectly** afterward → scene is *also*
  encoded nonlinearly/redundantly.

## 5. Conclusions

1. **The direction is not dead.** A real, cross-kitchen action signal exists in
   `v_t` and *survives* scene removal.
2. **But the recoverable granularity is coarse.** Fine 105-way verbs barely
   generalize across kitchens; coarse 13-way categories do. This is acceptable —
   coarse actions are the coarse end of the hierarchy we want.
3. **`v_t` is scene-dominated.** Naive VQ of raw `v_t` would produce a
   *kitchen*-flavored codebook, not an action one. **Scene factoring is on the
   critical path.**
4. **A linear projection is necessary but not sufficient** — it removes the linear
   scene component (for free, no action cost) but leaves a nonlinear scene
   component the MLP still exploits. Full de-scening needs a **structural or
   nonlinear** method.
5. **Clip length is not the lever.** 3 s clips don't beat ⅓ s at the honest
   splits; don't spend compute chasing longer clips.

## 6. Next steps (ranked)

1. **De-risk the temporal factoring (immediate, no training).** Test whether
   "scene = slow, action = fast" holds on our data: split each `v_t` into its
   **video-mean** (slow/scene part) and the **residual** (`v_t` − video-mean,
   fast/action part), then re-probe. Expect the video-mean to predict participant,
   and the residual to predict category *cross-kitchen* at least as well as raw
   `v_t` with less scene leakage. This is a linear stand-in for the two-codebook
   idea (§7); if it fails, the temporal-stability bias is wrong and §7 needs
   rethinking. Runs on the clip10 cache.

   → **Result (2026-07-13, `temporal_split.py`) — half-validated.** The
   **video-mean is a clean scene proxy** (decodes participant top1 1.0; cannot
   resolve within-video actions — category cross-kitchen macroF1 ~0.08, at
   majority). The **residual keeps the action** (category cross-kitchen macroF1
   **0.26** vs raw 0.25 — unchanged/slightly better) and **kills the LINEAR
   scene** (residual→participant *linear* top1 **0.18** ≈ chance). **But the
   residual still carries the NONLINEAR scene** — an MLP reads the kitchen from it
   at top1 **1.0**. Conclusion: action does live in the time-varying part and
   survives, but a subtractive/linear split cannot de-scene (same wall as the
   §4 factoring test). The §7 two-codebook plan therefore needs a **nonlinear /
   adversarial** disentangler, not mean-subtraction.
2. **Direct tokenization test.** VQ the (scene-reduced) `v_t` and compare
   **token↔category NMI vs token↔participant NMI** — does the codebook align with
   *actions* or *kitchens*?
3. **Learned projection + two-codebook disentangling (§7)** — the main proposed
   methodology; build only once 1–2 give a positive signal.
4. **(Optional) full clip30/clip90 extraction** to confirm the scale result on
   equal footing (equal clip count and kitchen coverage) before closing it.
5. Only after coarse action tokens look clean: **BPE composition** into a
   hierarchy and boundary-F1-across-levels vs matched-random — the headline
   claim. (Needs corpus-scale token statistics: run on all 153 videos.)

## 7. Proposed methodology (v2 direction — not yet built)

Motivated by the probe: the recoverable signal is coarse and scene-dominated, and
scene is *nonlinearly* encoded, so the representation itself must change — not
just a linear projection on top of avg-pooled features.

- **Replace avg-pool with a learned projection.** Mean-pooling the patch grid
  averages away within-clip motion — likely where much of the action lives. A
  small NN (MLP or transformer, TBD) projects the *unpooled patch features* into
  one lower-dim vector per clip: same downstream format, not naively pooled.
  Unlike avg-pool, a learned projection needs a training objective, so this is
  inseparable from the objective below.
- **Two codebooks: scene + action (factorized / disentangled).** Encode each clip
  into `z_scene` (VQ codebook 1) and `z_action` (VQ codebook 2), trained so the
  two components separate. Direct per-clip VQ has *no* pressure to isolate action
  from scene (it clusters whatever dominates = scene); the disentangling objective
  supplies that pressure — the same job the IDM/FDM does in a LAM, but yielding
  clean **per-clip** action tokens (stable within an action → BPE-friendly)
  instead of flickery transition tokens.
  - Label-free inductive bias: **scene ≈ constant within a video, action varies.**
  - Prior art: Denton & Birodkar 2017, "Unsupervised Learning of Disentangled
    Representations from Video" (content vs pose, adversarial separation).
- **Key risk — egocentric camera motion.** "Fast-changing = action" is
  confounded: head motion is fast and is *not* action. A naive slow/fast split
  would route camera motion into `z_action`. The factoring needs a handle to
  separate *action-motion* (local: hands/objects) from *camera-motion* (global:
  whole-frame). This is the appearance/camera-dominance pitfall and the main
  make-or-break for the two-codebook approach on this data.

---

## Appendix — reproduce

```bash
# clip10 (fully extracted)
python -m havq.analysis.gt     --config havq/analysis/configs/probe.yaml
python -m havq.analysis.probe  --config havq/analysis/configs/probe.yaml
python -m havq.analysis.factor --config havq/analysis/configs/probe.yaml
# clip90 (partial cache: participants P01–P05 only)
python -m havq.analysis.gt     --config havq/analysis/configs/probe_clip90.yaml
python -m havq.analysis.probe  --config havq/analysis/configs/probe_clip90.yaml
```

Outputs: `analysis_results/probe_clip10/{probe,factor}_report.json`,
`analysis_results/probe_clip90/probe_report.json`. Probe hyperparameters live in
the YAML configs (linear+MLP, 40 epochs, Adam lr 1e-3, batch 8192, z-scored).

---

## Changelog

- **v3 (2026-07-13)** — Ran the §6.1 de-risk test (`temporal_split.py`) and
  recorded the result inline: the video-mean/residual split is half-validated —
  action survives in the residual and the linear scene dies, but the nonlinear
  scene persists (MLP kitchen 1.0), so §7 needs a nonlinear/adversarial
  disentangler, not subtraction.
- **v2 (2026-07-13)** — Added §7 (proposed methodology: learned projection to
  replace avg-pool + two-codebook scene/action disentangling, with the
  egocentric camera-motion risk and Denton & Birodkar prior art). Reworked §6 to
  lead with the temporal-factoring de-risk test (video-mean vs residual).
  Clarified in §3 that verb is the *only* annotation — category is a
  deterministic coarsening and participant/kitchen is metadata.
- **v1 (2026-07-13)** — Initial probe report: assumptions, hypothesis, method,
  observations (clip10 + clip90), scene-factoring test, conclusions, next steps.
