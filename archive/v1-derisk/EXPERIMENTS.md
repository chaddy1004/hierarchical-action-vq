# Experiments

> Companion to `VQVAE_PIVOT_DESIGN.md`. That doc says *what* the pivot is; this
> doc enumerates *every experiment we could run to make it work* and the order to
> run them in. Grounded in the **v1 de-risk result** (2026-07-03), summarized
> below.

---

## Where we are (v1 de-risk result)

Full pipeline runs end-to-end (`features → gt → train → tokenize → eval`) on 3
HD-EPIC videos (P05/P06/P09) at `delta_min=10`, `H=1`, `|K|=256`, `latent=64`,
NSVQ.

- **Infra ✓** — V-JEPA loads offline, all six stages run, artifacts land in `results/`.
- **Model health ✓** — val MSE `1e-5` beats copy baseline `3e-5` (+59%); codebook
  perplexity 102/256 (no collapse).
- **Boundary signal ✗ (the blocker)** — tokens **flicker every clip** (mean run
  length = 1.0 → ~460 predicted boundaries vs ~73 GT). Model boundary
  F1@1s ≡ embedding-change F1 ≡ **random** F1 to 3 decimals (P05 .313, P06 .377,
  P09 .200). Recall saturates at 1.0, precision = base rate.
  **Root cause:** adjacent ⅓-s clips have cosine ≈0.99997 — each transition is a
  tiny noise-dominated residual, so the codebook spends codes on per-step noise
  with zero temporal persistence.
- **Not pure noise** — per-clip token↔verb NMI 0.27–0.48, purity 0.49–0.64. The
  tokens carry *clip-level* action content; the failure is temporal **instability**,
  not scene-flavoring.

**The one question every experiment below serves:** can we get a token stream
whose *changes* localize GT action boundaries better than random, without
sacrificing the action-meaningfulness we already have?

---

## How to read this

Each experiment lists:

- **Hypothesis** — what we think is true and why it should help.
- **Change** — the exact knob (`config.yaml` path) or code to touch.
- **Measure** — the metric that decides it (mostly `results/eval_report.json`).
- **Gate** — what result would make us keep / kill / escalate.

Priority tiers:

- **P0 — Unblock boundaries.** Nothing downstream matters until token changes
  beat random. Do these first.
- **P1 — Build & validate the hierarchy.** Only meaningful once a P0 config
  passes the boundary gate.
- **P2 — Meaning, naming, robustness, scale.** Strengthen the story / paper.

Cheap-first ordering inside each tier: readout changes (no retrain) before
retrains before backbone swaps.

---

## P0 — Unblock the boundary signal

The flicker has two independent cures: change the **temporal scale** so
transitions stop being noise (A, retrain), or change the **readout** so we stop
firing a boundary on every flip (B, no retrain). Do B first — it's free and
tells us whether the signal is already latent in the current tokens.

### Family A — Temporal scale (attacks the ≈0.99997 cosine at the source)

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **A1** | Coarser atomic clips make each embedding cover a real sub-action, so inter-clip cosine drops and transitions carry signal. | `features.delta_min` sweep: {30, 60, 90} (⅓s → 1s → 2s → 3s clips). Re-extract features. | Adjacent-clip cosine distribution; mean run length; boundary F1 vs random. | Keep any `delta_min` where model F1 > random F1 at ≥2 tolerances. |
| **A2** | A larger prediction horizon spans an action-scale transition instead of a frame-scale one. | `model.H` sweep {2, 4, 8} at fixed `delta_min`. Retrain + re-tokenize only (no re-extract). | Same, plus copy-baseline gap (larger H should make copy worse → more for the code to explain). | Keep H where F1 > random **and** model still beats copy. |
| **A3** | Subsampling N frames across a large window turns an unreliable 2-frame delta into a reliable clip summary (§5). | New encoder input: N equidistant clip embeddings across the window, not just `(v_t, v_{t+H})`. Touches `train.load_pairs`, `vq.NSVQ.f_enc` input dim, `tokenize_videos`. | F1 vs A2 at matched effective window. | Keep if it beats the 2-frame delta at the same window length. |
| **A4** | Overlapping clips give a denser, smoother token timeline that RLE can clean. | Add `features.stride` < `delta_min` (currently non-overlapping). Re-extract. | Mean run length; boundary F1; token stream smoothness. | Secondary — only if A1/A2 alone don't stabilize runs. |

### Family B — Boundary readout (no retrain; run against v1 tokens today)

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **B1** | Flicker is high-frequency; a boundary should require a *sustained* token change. | In `eval`, mode-smooth the token stream (sliding-window majority vote, window w∈{3,5,7}) **then** diff. | F1 vs random at each w. | If smoothing lifts F1 above random, the signal was there — flicker was a readout artifact. |
| **B2** | We over-predict; a budget-matched top-K readout is the fair test. | Rank change positions by a confidence score (codebook argmin **margin**, or encoder-delta magnitude `‖d‖`); take top-K with K = |GT| and K = |GT|·{1,2}. | Precision at matched budget; F1. | Beats the same-budget random baseline → real localization. |
| **B3** | Boundaries live in the continuous delta, not the discretized token. | Predict boundaries from peaks of `‖d‖` or `1−cos(v_j,v_{j+H})` with NMS (stepsegmenter-style), ignoring token identity. | F1 vs the token-change readout. | Diagnostic: isolates whether quantization *helps* or *hurts* boundary localization. |
| **B4** | Min-run-length RLE removes singleton runs before scoring. | RLE the stream, drop runs < m clips (merge into neighbor), m∈{2,3,4}, then boundaries = run edges. | Run-count vs |GT|; F1. | Same gate as B1; complementary knob. |

### Family C — Bottleneck / codebook capacity (force codes onto structure, not noise)

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **C1** | A smaller codebook can't afford to spend codes on per-step noise → coarser, more persistent tokens. | `model.codebook_size` sweep {16, 32, 64, 128, 256, 512}. Retrain. | Mean run length; perplexity; F1. | Look for the knee where run length > 1 without recon collapse. |
| **C2** | A tighter latent bottleneck drops noise dimensions of the delta. | `model.latent_dim` sweep {8, 16, 32, 64}. Retrain. | val MSE vs run length trade-off. | Keep the smallest latent that still beats copy. |
| **C3** | NSVQ vs EMA-codebook changes what survives quantization (§2). | Add an EMA-VQ variant of `vq.NSVQ`; compare head-to-head. | F1, perplexity, dead-code count. | Keep whichever gives more persistent, higher-F1 tokens. |
| **C4** | Dead-code replacement cadence affects how many codes chase noise. | `nsvq.replace_dead_codes_every` / `replacement_warmup_batches` sweep. | perplexity trajectory; final F1. | Minor tuning; only if C1/C2 are promising. |

### Family D — Input representation

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **D1** | Raw (un-normalized) features may preserve magnitude cues L2-norm destroys. | Toggle `l2_normalize` in `train`/`tokenize`/`eval`. | F1; recon MSE scale. | Diagnostic — pick whichever localizes better; keep norm if tie (repo convention). |
| **D2** | Feeding the *difference* `v_{t+H}−v_t` (not the pair) concentrates the transition. | Encoder input = concat(`v_t`, `v_{t+H}−v_t`) or just the delta. | F1; run length. | Keep if it beats the plain pair. |
| **D3** | V-JEPA deltas may be appearance-dominated; a different backbone may carry more action. | Swap backbone: **V-JEPA 2 action-conditioned** variant, or **LaVILA** features (§2). Re-extract. | F1; token↔verb NMI. | Escalation path if A–C plateau below random. Expensive — last resort. |

**P0 exit gate:** at least one config produces **model boundary F1 > random F1**
at the primary tolerance on ≥2 of 3 videos, with the model still beating the
copy baseline. Until this passes, do **not** start P1.

---

## P1 — Build & validate the hierarchy (Design B, the primary paper path)

Only run these once a P0 config clears the boundary gate.

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **E1** | RLE turns the (now-stable) token stream into clean atomic `LeafNode` segments (§4.1). | New `havq/hierarchy.py`: RLE → leaves. | Leaf count vs finest GT granularity; boundary F1 at leaf level. | Leaves should approximate the finest GT level. |
| **E2** | BPE-style adjacent-pair merging yields a dendrogram whose cuts trace the GT granularity curve (§4). | Agglomerative merge by corpus adjacent-pair frequency; record full merge tree; emit F1 at every cut. | **Boundary-F1 curve across levels** (the count-free headline). | Curve should touch/beat stepsegmenter's divisive tree at matched levels. |
| **E3** | Codebook-vector similarity is a better linkage than raw frequency. | Alt linkage = cosine of adjacent segment code vectors; compare to E2. | F1 curve, both linkages. | Keep the better; report both as ablation. |
| **E4** | MDL gives a single "natural" granularity for a table number (§4.4, optional). | Description-length stopping on the merge tree. | Chosen level's F1 vs GT's own N. | Non-load-bearing; nice-to-have. |
| **F1** | Multi-scale H (Design A fallback): stacking per-H streams gives a hierarchy without BPE (§5). | Train tokenizers at H∈{2,8,32}; stack coarse-on-fine. | Per-level F1; level-nesting violations. | Baseline-to-beat for E2; keep as ablation even if BPE wins. |
| **F2** | Subsampled **clip tokens** at large H represent activity spans a 2-frame delta can't (§5). | A3 machinery at large H for the coarse levels of F1. | Coarse-level F1. | Rescues large-H coarse tokens if F1 large-H two-frame degrades. |

---

## P2 — Meaning, naming, robustness, scale

Can run in parallel with P1 once P0 passes; strengthens the paper, doesn't gate it.

### Action-meaningfulness & nameability (§6, §7)

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **G1** | Tokens carry action identity (already NMI 0.27–0.48 at clip level). | Recompute purity/NMI on the *P0-winning* config. | token↔verb purity, NMI. | Should rise with more persistent tokens. |
| **G2** | Directional actions get **different** tokens (open vs close drawer, take vs put). | Controlled probe: pull GT segments of paired inverse verbs; check token distributions differ. | Token-distribution divergence between inverse verbs. | Passing = tokens are action- not scene-meaningful (§9 core question). |
| **G3** | A camera pan / no-action span does **not** spuriously get its own token. | Probe low-motion / background (`label=-1`) spans. | Token entropy over background spans. | Passing = not scene-dominated. |
| **G4** | Post-hoc lexicon: aggregate VLM captions per token index → nameable primitives (§6.ii). | Caption RLE segments, aggregate by token id. | Dominant-caption purity per token. | Upside measurement; feeds the "discovered vocabulary" claim. |
| **G5** | If G1/G4 show entanglement, train-time language alignment fixes it (§6). | Contrastive loss pulling codes toward caption embeddings. | F1 + purity vs prediction-only. | **Only** if post-hoc proves tokens un-nameable. Escalation. |

### Robustness & scale

| ID | Hypothesis | Change | Measure | Gate |
|----|-----------|--------|---------|------|
| **H1** | Results hold beyond 3 videos. | Scale to the full annotated HD-EPIC set. | F1 mean ± std across videos. | Stability of the P0 finding. |
| **H2** | The tokenizer generalizes; it's not per-video overfit. | Train on a video subset, tokenize held-out videos. | Held-out F1 vs in-train F1. | Small gap = the codebook is a real shared vocabulary. |
| **H3** | Findings aren't a lucky seed. | `train.seed` sweep {0,1,2}. | F1 variance across seeds. | Report as error bars. |
| **H4** | We beat the old stepsegmenter Stage-1 boundary cue. | Run change-score + NMS baseline on the same videos. | F1: VQ tokens vs change+NMS. | This is the headline comparison the pivot must win. |

---

## Recommended first sequence

1. **B1 + B2 + B4** on the existing v1 tokens (free, today). If any lifts F1
   above random, the flicker was a readout artifact and P0 is nearly solved.
2. **A1** (`delta_min` sweep) — the most likely structural fix; re-extract once,
   reuse across later runs.
3. **A2** (`H` sweep) on the best `delta_min`.
4. **C1** (`codebook_size`) on the best (delta_min, H).
5. Re-check the **P0 exit gate**. If it passes → start **E1/E2**. If it plateaus
   below random after A+B+C → try **D2**, then escalate to **D3** (backbone swap).

Log every run's config + `eval_report.json` so the sweep is reconstructable.
