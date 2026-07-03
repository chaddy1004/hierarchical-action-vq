# StepSegmenter — VQ-VAE Pivot Design Doc

> **Status:** design / brainstorm output, pre-implementation.
> **Purpose:** self-contained handoff for an implementation LLM that has NOT seen the brainstorming conversation. Read this end-to-end before writing code. It captures a proposed **pivot of StepSegmenter's methodology from a GNN-based approach to a VQ-VAE (latent-action) approach**, inspired by LAPA (Latent Action Pretraining, ICLR 2025, arXiv:2410.11758).
>
> The goal of the first coding effort is **not** the full system — it is a small **de-risking experiment** (see §9). Do that first.

---

## 0. Context you need (StepSegmenter in one paragraph)

StepSegmenter does **vocabulary-free, count-free temporal action segmentation (TAS)** on procedural videos (cooking, assembly, egocentric manipulation). "Vocabulary-free" = it never consumes the benchmark's action-label set `C`. "Count-free" = it never consumes the oracle number of segments `N`. Instead it produces a **full hierarchy** spanning multiple granularities and is evaluated with **boundary-F1 at every level** — the claim being that the annotator's chosen granularity is *contained* in the hierarchy without ever being given `N`. The current (pre-pivot) pipeline is: V-JEPA clip embeddings → change-score boundaries + NMS → VLM captions → semantic embedding → recursive spectral clustering into a tree. There was a GNN thread (refine features with a graph) that this pivot **replaces**.

**Repo conventions that matter for code:** use `os.path` (never `pathlib`); all frame arrays `np.ndarray` `T×H×W×C` uint8; all embeddings L2-normalized before similarity. Key types live in `stepsegmenter/utils/graph_types.py` (`LeafNode`, `InternalNode`, `HierarchicalActionGraph`).

---

## 1. The core idea of the pivot

LAPA showed you can extract **discrete "latent actions" from unlabeled video** with a VQ-VAE: encode the transition between two frames `x_t` and `x_{t+H}` into a discrete code `z_t` drawn from a learned codebook, trained purely by reconstructing `x_{t+H}` from `x_t` + `z_t`. No action labels, no vocabulary.

We adapt this to TAS. The pivot has three moves:

1. **Tokenize the video into a stream of discrete action codes** using a VQ-VAE-style tokenizer.
2. **Build the granularity hierarchy by BPE-style composition** of that token stream (primary path), with a multi-H approach as a fallback.
3. **Attach natural language** to the resulting segments via captioning (post-hoc lexicon).

This keeps StepSegmenter's crown-jewel framing (vocabulary-free, count-free, hierarchy-as-evaluation) fully intact — arguably strengthens it — while swapping the mechanism.

**Training is acceptable.** The user has compute and is not attached to the previous "training-free" contribution. (If training-free is ever wanted again, freeze everything and use LAPA's released checkpoint — but see §2 for why we avoid that.)

---

## 2. Key architectural decision: train ONLY a codebook on top of a frozen foundation model

**Do NOT use LAPA's released checkpoint.** It was trained on robot manipulation video (Bridge, Open-X, Something-Something). The domain gap to procedural/cooking/egocentric benchmarks makes it risky and an unfair comparison.

**Do NOT train a full pixel-reconstruction VQ-VAE from scratch** (LAPA-scale: ~300M params, 8×H100×34h). Too expensive.

**Instead: freeze a video foundation model (V-JEPA) and train only a small quantization module on top of its features.** V-JEPA is a *Joint Embedding Predictive Architecture* — it is natively designed to **predict in embedding space, not pixel space**. So we predict the next *embedding*, not the next frame:

```
v_t     = VJEPA(clip @ t)          # frozen, no grad
v_{t+H} = VJEPA(clip @ t+H)         # frozen, no grad
d       = f_enc(v_t, v_{t+H})       # small trainable "delta" encoder (e.g. MLP)
z, idx  = Quantize(d)               # trainable codebook (VQ). idx = integer token
v_hat   = f_dec(v_t, z)             # small trainable predictor
loss    = || v_{t+H} - v_hat ||^2   # predict the next EMBEDDING (+ VQ commitment loss)
```

Only `f_enc`, the codebook, and `f_dec` are trained. Everything heavy (perception) is frozen V-JEPA. This is cheap and philosophically native to V-JEPA.

**Notes / choices for the implementer:**
- Use standard VQ-VAE machinery (codebook + commitment loss + straight-through estimator). Consider **NSVQ** (noise-substitution VQ, used by LAPA) or EMA codebook updates to avoid codebook collapse. Also consider codebook re-initialization of dead entries.
- **Codebook size `|K|`** (e.g. 256 or 512) is the *atomic action vocabulary size*. It is a hyperparameter and is **NOT** the benchmark's `|C|` and **NOT** the per-video segment count `N`. Staying vocabulary-free/count-free is preserved.
- Alternative frozen backbones worth trying: **V-JEPA 2** (has an action-conditioned world-model variant — even closer fit; check what's released), or **LaVILA/NARRATOR visual features** (would revive the old "single feature lineage" idea in a cleaner form). V-JEPA is the natural first choice.
- **Known risk:** V-JEPA deltas may be dominated by *appearance/camera* change rather than *action*. The codebook will quantize whatever structure exists — must verify empirically that tokens are action-meaningful, not scene-meaningful (this is exactly what §9 tests).

---

## 3. Terminology (so the rest of the doc is unambiguous)

- **Codebook** = fixed list of `|K|` learned vectors, each with an integer index `0..|K|-1`. This is the *alphabet*.
- **Token** = one codebook **index** assigned to one position in the video (the "action code"). Token = code = codebook entry index — same object. Each entry has two faces: the **integer index** (used as a discrete symbol for BPE/boundaries) and the **vector** it points to (used for continuous similarity if needed).
- **Token stream** = the sequence of tokens along the video timeline, e.g. `[12, 12, 12, 45, 45, 3, 3, 78, ...]`. **This turns the video into a string of symbols, structurally identical to text.** This is automatic the moment you quantize a whole video (it is *not* an extra invented step). LAPA produces such tokens but consumes them one-at-a-time for robot control; it never streams+segments them. **Streaming + compressing them into a hierarchy is our novel contribution.**
- **Small H** → **delta token** ("what changed", directional, atomic).
- **Large H** with N subsampled frames → **clip token** ("what is happening in this span", activity-centric). (See §5 on subsampling.)
- **Coarse granularity = large H = fewer, more general segments.** Fine granularity = small H = many atomic segments.

---

## 4. Hierarchy — PRIMARY PATH: Video BPE (Design B)

Train **ONE** tokenizer at small H → get **one** fine token stream per video → build the hierarchy by **BPE-style agglomerative composition** of that single stream.

BPE = repeatedly merge the most frequent **adjacent** token pair (measured across the corpus) into a new symbol. Each merge = one step up the hierarchy.

```
atomic (0 merges):    [reach][grasp][lift][reach][grasp][pour][stir]   7 segments  (finest)
merge (reach,grasp):  [ pick ][lift][ pick ][pour][stir]              5 segments
merge (pick,lift):    [pick-up][ pick ][pour][stir]                   4 segments
...
final (max merges):   [ whole activity ]                              1 segment   (coarsest)
```

**Critical clarification (this was a point of confusion):** BPE is **not** a binary done/not-done algorithm. It produces an **ordered sequence of merges**, i.e. a **dendrogram (merge tree)**. Merges are pairwise, but the *outcome* is a full continuum of granularities: **granularity = how many merges you have applied / where you cut the dendrogram.** Every intermediate cut is a valid hierarchy level "in the middle."

**This is the SAME evaluation StepSegmenter already does.** The current method builds a tree top-down (divisive spectral clustering) and reports boundary-F1 at every depth. BPE builds the same kind of tree **bottom-up (agglomerative)** and reports F1 at every cut. Identical eval mechanism, opposite construction direction. Count-free contribution is preserved.

**Steps to implement:**
1. **RLE the token stream** into initial atomic segments (a run of identical/compatible tokens = one leaf). This is the step that converts per-position tokens into `LeafNode`s. (Delta tokens are "pairwise" — they describe transitions — so RLE / boundary-on-change is what recovers segments. Clip tokens are closer to segment labels directly.)
2. **Agglomerative merge with a temporal-contiguity constraint** (only adjacent segments may merge). Linkage criterion = corpus-frequency of the adjacent token/pattern pair (classic BPE), OR codebook-vector similarity of adjacent segments (agglomerative-clustering style). Try both.
3. **Record the full merge tree.** Do NOT stop early for the hierarchy eval — emit all levels and report the F1 curve.
4. **(Optional) MDL / compression stopping** only if a single canonical "natural" granularity is needed for a table number. Not load-bearing; the hierarchy eval reads all levels regardless.

---

## 5. Hierarchy — FALLBACK / ABLATION: Multi-scale H (Design A)

**Keep this — the user explicitly wants it retained as a fallback/ablation, not thrown away.** It is the safer, more obvious route and a legitimate baseline-to-beat.

Train **several** tokenizers, one per window size H (e.g. H ∈ {2, 8, 32}). Each labels the video into a token stream at that granularity. Coarser H → slower-changing stream → coarser boundaries. Hierarchy = **stack the streams**, coarse on top.

**Its known weaknesses (which is why it's the fallback, and why it's a good baseline to beat):**
- **Levels are not guaranteed to nest** — a coarse-H boundary may not land on a fine-H boundary, so forcing a clean tree requires reconciliation.
- **Tokens across models are not comparable** — each H trains its own codebook, so model-H2's token #37 is unrelated to model-H8's token #37. Levels can only be related by time-overlap, not token identity.

**Subsampling trick (belongs here, and rescues large H):** for large H, don't feed just two frames — subsample N equidistant frames across the window (every H/N frames) and encode them. This gives the tokenizer the intermediate trajectory, turning an unreliable two-frame *delta* into a reliable multi-frame *clip summary*. This is what makes "large H = high-level instruction" actually viable (a two-frame delta over a 30s span cannot represent "make coffee"; a subsampled clip can). LAPA's own appendix notes large-H two-frame reconstruction degrades — subsampling is the fix. (For small H there aren't enough frames to subsample; that's fine, small H stays delta-like.)

---

## 6. Attaching natural language

There are **two distinct linking questions**. Only the first is required.

**(i) Label the SEGMENTS the hierarchy outputs — REQUIRED, easy, no guarantees needed.**
A segment (token-run or BPE-merged unit) is a real video clip with start/end frames. Caption *that clip* with the VLM — **exactly what the current pipeline already does**. This needs zero guarantee about token↔language cleanliness because you describe the actual footage. The labeled hierarchy is therefore never at risk.

**(ii) Give each TOKEN a canonical meaning (the "lexicon") — OPTIONAL, and it is a measurement.**
Aggregate captions per token index across the whole corpus: every place token #37 appears → collect its segment's caption → the dominant theme is #37's meaning (e.g. 200 instances mostly "pour ..." ⇒ #37 = "pour liquid"). Benefits: averages out single-clip caption errors (e.g. direction flips like open/close, take/put), and gives each atomic action a stable identity. Critically, it **measures how nameable the tokens are** = codebook purity (see §7). This is upside, not a dependency.

**Do tokens cleanly equal "reach"/"grasp"/etc.? No guarantee.** The codebook is trained for *prediction error*, not language alignment. Tokens may be polysemous, fragmented, or entangled with appearance/camera motion. Two strategies:

| | Force language alignment (train-time) | Label post-hoc (the lexicon) |
|---|---|---|
| How | contrastive loss pulling codebook entries toward caption embeddings; or a language-prediction auxiliary head | train tokenizer for prediction only; caption + aggregate afterward |
| Tokens clean by | construction | luck (then measured) |
| Cost / risk | needs captions during training; can hurt action fidelity; couples to captioner | cheap; tokenizer stays unbiased |
| Diagnostic? | no (forced) | **yes — tells you if tokens are nameable** |

**Recommendation: post-hoc first (diagnose, then treat).** Run the lexicon, look at purity. If tokens are cleanly nameable, done. If hopelessly entangled, *then* add train-time language alignment as the fix. Don't pay the train-time cost/risk until the measurement proves it's needed. Either way, (i) still yields a fully labeled hierarchy.

---

## 7. Evaluation

- **PRIMARY (keep):** boundary-F1 + frame-accuracy (Hungarian) on the TAS benchmarks, reported **across all hierarchy levels** (the count-free curve). This keeps comparability to prior work and preserves the hierarchy-as-evaluation contribution. Existing eval code: `stepsegmenter/evaluation/` (`baseline.py`, `hungarian_eval.py`), plus stubs in `metrics.py`.
- **SECONDARY (new, bonus — showcases the discovered codebook):**
  - **Codebook purity / action-primitive quality:** mutual information or purity between token indices and GT action labels ("does token #37 consistently map to one real action?"). This is also the diagnostic from §6(ii).
  - **Compression / MDL** numbers from BPE (evidence the hierarchy is an efficient description).

Boundary-F1 stays the headline; codebook purity + compression are supporting evidence, not replacements.

---

## 8. What survives / what changes vs. the old pipeline

| Component | Fate |
|---|---|
| Count-free / vocabulary-free framing | **Survives, strengthened** (codebook = discovered vocab; BPE/MDL = count-free) |
| Hierarchy-as-granularity evaluation | **Survives unchanged** (best contribution) |
| Boundary-F1 across levels | **Primary metric, unchanged** |
| V-JEPA change-score + NMS Stage 1 | **Replaced** by VQ tokenizer on frozen V-JEPA features |
| Recursive spectral clustering / eigengap | **Replaced** by BPE agglomerative composition (or Design-A stacking) |
| GNN thread (old Phases 3–7) | **Dropped** — this pivot supersedes it |
| Captioning / semantic merge | **Repurposed** as segment captioning + token→caption lexicon |
| "Training-free" contribution | **Dropped** (acceptable to user); recoverable only via frozen LAPA, which we avoid |

---

## 9. FIRST EXPERIMENT (do this before any architecture commitment)

Everything above stands or falls on one empirical question: **do V-JEPA-derived discrete tokens align with real action structure, and are they action-meaningful rather than scene-meaningful?**

**Minimal de-risking experiment:**
1. Take 3–5 benchmark videos (with GT action boundaries/labels available).
2. Extract frozen V-JEPA clip embeddings along each video at a small H.
3. Train a **tiny** VQ module (§2) — codebook (e.g. |K|=256) + small encoder/decoder — to predict `v_{t+H}` from `v_t + z`. This is small and fast; a short training run suffices for a first look.
4. Produce the **token stream** for each video and **look at it**:
   - Do token *changes* line up with GT action boundaries? (overlay token-change positions vs GT boundaries)
   - Are token-runs stable and RLE-able into clean segments, or is it high-frequency noise?
   - Are tokens **action-meaningful vs scene-meaningful**? (e.g. does open-drawer vs close-drawer get *different* tokens? does a camera pan with no action spuriously get its own token?)
   - Quick purity check: token index vs GT label — is there any signal?

**Decision gate:**
- **If tokens align and are action-meaningful** → the whole Design-B (Video BPE) paper is viable and cheap. Proceed to RLE → BPE hierarchy → segment captioning.
- **If tokens are noisy or scene-dominated** → try: subsampled clip tokens (§5), a different frozen backbone (V-JEPA 2 / LaVILA), or train-time language alignment (§6). If still bad, Design A (multi-scale H) is the fallback.

Do **not** build the full BPE hierarchy, lexicon, or eval harness until step 4 shows the token stream carries action signal.

---

## 10. Open risks (keep visible)

1. **V-JEPA delta may encode appearance/camera, not action.** (Tested in §9.)
2. **Domain transfer** of any frozen backbone to the target benchmark is unproven.
3. **Delta tokens are pairwise (transitions), not segments** — RLE/boundary step needed to recover segments; may be noisy. Clip tokens (subsampling) mitigate.
4. **Discrete "similarity"** doesn't map onto cosine like caption embeddings did — BPE uses token identity/frequency or codebook-vector similarity; different machinery than spectral clustering.
5. **Token nameability is not guaranteed** — mitigated by post-hoc lexicon + optional train-time alignment.

No perfect idea exists; these are the accepted risks of the pivot. The §9 experiment is designed to kill the biggest one first.

---

## 11. Reference

LAPA — *Latent Action Pretraining From Videos*, ICLR 2025, arXiv:2410.11758v2. VQ-VAE over frame pairs `(x_t, x_{t+H})` → discrete latent action `z_t` (codebook size `|C|`); NSVQ to avoid collapse; cross-attention decoder reconstructs `x_{t+H}`; latent actions later used for VLA behavior cloning. Note: LAPA reconstructs **pixels**; we reconstruct **embeddings** on frozen V-JEPA. LAPA invokes the BPE analogy in its intro but **never builds a compositional hierarchy** — that is the white space this pivot occupies.

