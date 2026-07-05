# PLAN
w si

## Research goal

**Vocabulary-free, count-free temporal action segmentation via discrete latent
actions.** Turn a video into a stream of discrete "action tokens" — a small VQ
model trained on frozen V-JEPA features, LAPA-style (the token encodes the
*transition* between clips, not the clips themselves). Then build the
multi-granularity segmentation hierarchy by BPE-style merging of that token
stream, and report boundary-F1 at every level of the hierarchy.

**The claim that makes this publishable:** LAPA extracts latent-action tokens
but consumes them one at a time for robot control. *Streaming* the tokens and
*composing* them into a segmentation hierarchy is unclaimed. The codebook is a
discovered vocabulary (vocabulary-free) and every dendrogram cut is a
granularity level (count-free) — StepSegmenter's framing, new mechanism.

## What v1 taught us (archived in `archive/v1-derisk/`)

The plumbing works and the VQ model trains healthily, but at ⅓-second clips the
transition tokens **flicker every step** (adjacent-clip cosine ≈ 0.9999 — the
transitions are noise), so naive token-change boundaries tie random. The tokens
do carry action content (token↔verb NMI 0.27–0.48). Two things were never
established: **the right temporal scale**, and **whether BPE composition works
at all**. That's exactly what the two experiments below test — one each.

## Experiment 1 — Find the temporal scale where the signal lives (no training)

**Question:** at what clip length do frozen V-JEPA features actually *see*
action boundaries?

**Method:** on the 3 annotated test videos only. Extract clip features at
`delta_min` ∈ {10, 30, 90} frames (⅓ s, 1 s, 3 s). No VQ, no training. For each
scale measure:
1. the adjacent-clip cosine distribution (is there anything but noise?), and
2. boundary F1 of the top-K embedding-change peaks (K = number of GT
   boundaries) vs. matched-count random placement.

**Decision:** pick the scale where raw feature change beats random most
clearly. If **no** scale beats random, the backbone/domain is the problem and
no tokenizer on top can fix it — stop and rethink the backbone before anything
else.

**Cost:** ~1 GPU-hour, one afternoon.

## Experiment 2 — Does discretize + BPE-compose recover a hierarchy? (the hypothesis)

**Question:** does BPE-merging the token stream produce *some* granularity
level whose boundaries beat random? This is the first-ever test of the paper's
actual claim — v1 only tested the strawman "boundary at every token flip."

**Method:** at the winning scale from Exp 1: train the small NSVQ tokenizer
(reuse `archive/v1-derisk/havq/` as the starting point), tokenize the 3 videos,
then: RLE the stream into runs → repeatedly merge the most frequent adjacent
token pair (recording the full merge tree) → boundary-F1 at **every** cut of
the tree vs. matched-count random.

**Decision:** any cut beating random on ≥2 of 3 videos → the angle is alive;
scale up to the full dataset (153 videos, ~40 h at `~/datasets/HD-EPIC/Videos/`)
and make the F1-across-levels curve the headline figure. No cut beats random →
the tokens lack recurring temporal structure; revisit tokenization (clip tokens
instead of transition tokens, larger codebook context) before touching BPE
again.

## Explicitly NOT now

Full-dataset extraction, codebook/latent-size sweeps, multi-H hierarchies,
smoothing tricks, captions/lexicon. All deferred until Exp 2 has a verdict.
(The long lever list lives in `archive/v1-derisk/EXPERIMENTS.md`.)
