# RESEARCH.md

The research framing for this repo: what we're claiming, why it might be
true, what usually kills projects like this, and what we must read before
claiming novelty. Operational details (which experiment runs next) live in
`PLAN.md`; this file is the slow-moving context behind them.

---

## Main idea

**Treat a video as a sentence in an unknown language and discover its
structure the way NLP does: tokenize, then compose.**

1. **Tokenize** — a small VQ model on top of a *frozen* video foundation model
   (V-JEPA) turns each temporal transition into one discrete code from a
   learned codebook (LAPA-style: encode the pair (v_t, v_{t+H}), reconstruct
   v_{t+H} from v_t + code, so the code is forced to carry the *change* — the
   action — not the scene). The video becomes a stream of integer tokens.
2. **Compose** — merge the token stream bottom-up, BPE-style (repeatedly merge
   the most frequent adjacent pair across the corpus, recording the full merge
   tree). Every cut of the resulting dendrogram is a segmentation at one
   granularity; the tree *is* the multi-granularity output.

This yields temporal action segmentation that is **vocabulary-free** (the
codebook is discovered, never the benchmark's label set) and **count-free**
(no segment count is consumed; all granularities are emitted and evaluated).

**The novelty claim:** latent-action tokenizers exist (LAPA, Genie, LAPO), but
they consume tokens one-at-a-time for control. *Streaming* the tokens over
long video and *composing* them into a segmentation hierarchy is — per the
2026-07-05 sweep in `RELATED_WORKS.md` — still unclaimed **in this exact
form**, but the neighborhood is crowded and closing fast; three papers each
hold one piece of it.

## Main hypothesis

> Discrete latent-action tokens extracted from a frozen video foundation
> model contain enough recurring temporal structure that frequency-based
> composition (BPE) recovers a granularity hierarchy aligning with human
> action annotations — without labels, vocabulary, or segment counts.

Decomposed into testable parts (status from the v1 de-risk run, archived in
`archive/v1-derisk/`):

- **H1 — Action content:** frozen-backbone transitions can be discretized
  into tokens that correlate with real actions, not just scenes.
  *Status: partial support — token↔verb NMI 0.27–0.48, purity 0.49–0.64 at
  clip level.*
- **H2 — Temporal structure:** the token stream has persistence and recurring
  patterns (runs, frequent adjacent pairs) rather than being i.i.d. noise in
  time. *Status: FAILED at ⅓-second clips (tokens flicker every step; mean
  run length 1.0). Untested at coarser temporal scales — this is Exp 1.*
- **H3 — Compositional alignment:** cuts of the BPE merge tree beat matched
  random baselines on boundary F1 at some granularity. *Status: never tested —
  this is Exp 2 and the core of the paper.*

H1 without H2 = meaningful but unstable tokens (v1's exact situation).
H2 without H3 = clean segments that don't align with human granularity.
The paper needs all three.

## Common pitfalls (each with its guard)

**Representation**

1. **Temporal scale mismatch** *(confirmed in v1)*: too-fine clips make
   adjacent embeddings nearly identical (cosine ≈0.9999), so transitions are
   noise and the codebook quantizes noise. Guard: measure the adjacent-cosine
   distribution and raw-feature boundary signal *before* training anything at
   a given scale (Exp 1 exists because of this).
2. **Appearance/camera dominance**: deltas may encode scene change, head
   motion (egocentric!), lighting — not action. Guard: probe pairs of inverse
   actions (open vs close should get *different* tokens) and no-action camera
   motion spans (should NOT get their own stable token); report token↔verb
   NMI, not just reconstruction loss.
3. **Trivial decoder**: if the decoder can reconstruct v_{t+H} from v_t alone,
   the code carries nothing. Guard: always report the copy baseline
   (v̂ = v_t) and an ablation with z shuffled/zeroed.

**Quantization**

4. **Codebook collapse** (perplexity → 1) or the dual failure, codes spent on
   per-step noise. Guard: track perplexity + mean run length together; healthy
   is high-ish perplexity AND runs ≫ 1.
5. **|K| quietly becomes the vocabulary prior**: tuning codebook size per
   benchmark until numbers look good undermines the vocabulary-free claim.
   Guard: fix |K| globally, or show robustness across a range.

**Composition**

6. **BPE needs corpus-scale statistics**: pair frequencies from a handful of
   videos are noise. v1 trained on 4 videos / 3.7k pairs — far too thin.
   Guard: run composition on the full corpus (~40 h HD-EPIC), not test snippets.
7. **BPE merges by token identity, not similarity**: flickering alternation
   (a,b,a,b) only merges if the *exact pair* recurs corpus-wide; near-miss
   codes stay separate. Guard: also try codebook-vector-similarity linkage;
   consider RLE before BPE.
8. **Transition tokens are pairwise, not segments**: a token labels a
   *boundary between* clips; recovering segments needs RLE/run logic, and
   off-by-one conventions bite. Guard: fix the index convention in writing
   (token j = transition j→j+H) and unit-test the mapping.

**Evaluation**

9. **Recall saturation** *(bit us in v1)*: over-predicting boundaries gives
   recall 1.0 and an F1 that looks nonzero but equals the random baseline.
   Guard: every F1 ships with a matched-count random baseline and a
   same-budget strong baseline (raw feature-change peaks); report
   precision/recall, not just F1.
10. **Cherry-picked tolerance / granularity level**: with a whole dendrogram
    of levels and several tolerances, *something* will look good. Guard:
    pre-register the primary metric (which tolerance, how levels are chosen);
    report full curves.
11. **Leakage into "unsupervised"**: tokenizer hyperparameters tuned on eval
    videos, or the eval set used to pick the granularity cut. Guard: tune on
    held-out videos; the count-free claim means the *method* never selects a
    single level using GT.
12. **Unfair comparisons**: most unsupervised TAS baselines consume N
    (segment count) or C (label count). Comparing without stating that
    invites rejection. Guard: a supervision-consumed table (method vs. what
    it's given) in the paper.

## Literature review that must be done

Grouped by the question each answers. Priority = ★.

**A. Is the novelty claim actually true? (do this first)**

- ★ **LAPA** (Ye et al., ICLR 2025) — the tokenizer we adapt; know its NSVQ
  details and its appendix on large-H degradation.
- ★ **Genie** (Bruce et al., 2024) — latent-action VQ codebook from video at
  scale; check they never segment/compose.
- **LAPO** (Schmidt & Jiang, ICLR 2024), **IGOR** (Microsoft 2024), **Moto**
  (2024) — latent-action-from-video family; same check.
- ★ **Zero-resource speech / unit discovery**: textless NLP (HuBERT units →
  language modeling), ZeroSpeech challenges, unsupervised word segmentation
  from phone-like units (Goldwater's Bayesian segmentation, adaptor grammars,
  Morfessor/MDL). *This is the closest existing "discrete units + composition
  → discovered lexicon" pipeline — if video-BPE has an analog anywhere, it's
  here, and reviewers from speech will know it. Also a source of better
  composition algorithms than vanilla BPE.*
- BPE-on-nontext generally: byte/unit BPE for speech units, music tokens
  (e.g. MIDI BPE), protein sequences — cite as analogy, check none do video.

**B. What are we competing against? (baselines + protocols)**

- ★ Unsupervised TAS: CTE (Kukleva 2019), TW-FINCH (Sarfraz 2021), ABD,
  TOT/UDE-style recent work; note exactly what each consumes (N? C? features?).
- ★ TAS surveys (e.g. Ding et al. 2023) for the standard benchmarks, metrics
  (boundary F1 tolerances, Hungarian frame accuracy), and protocol details.
- Generic event boundary detection (GEBD, Shou et al. 2021) — boundary-only
  competitor line; their predictability-based boundary signal is also a
  candidate baseline cue.
- StepSegmenter's own prior baselines (change-score + NMS) — the in-house
  bar to beat.

**C. What's known about the backbone features?**

- ★ V-JEPA / V-JEPA 2 papers — what the features encode, the
  action-conditioned variant (V-JEPA 2-AC) as an alternative backbone.
- LaVILA / egocentric video-language features — backup backbone with more
  action-language grounding.
- Any probing/analysis work on what video-JEPA features capture
  (appearance vs motion) — directly informs pitfall 2.

**D. Datasets & their annotation granularity**

- ★ HD-EPIC (current data), EPIC-KITCHENS-100, Ego4D — narration timestamp
  conventions, verb taxonomies, known annotation quirks.
- Breakfast / 50Salads / GTEA / EgoProceL / Assembly101 — the TAS standards a
  paper will be expected to report on; note which have multi-granularity
  annotations (Assembly101, FineGym) for evaluating the *hierarchy* claim.

**E. Machinery (read as needed)**

- VQ-VAE (van den Oord 2017), NSVQ (Vali & Bäckström 2022), FSQ (Mentzer
  2023 — simpler quantizer, drop-in candidate), codebook-collapse remedies.
- Hierarchical/agglomerative segmentation evaluation: dendrogram purity,
  hierarchy metrics — how to score a tree against flat GT fairly.
