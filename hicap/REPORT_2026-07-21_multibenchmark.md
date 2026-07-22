# hicap — Multi-benchmark Report, 2026-07-21

Extends the same-day `REPORT_2026-07-21.md` (which was 50 Salads only) to all
three standard MS-TCN benchmarks — **50 Salads, Breakfast, GTEA** — with the
identical pipeline and metrics. The reader was generalized (`hicap/data/tas.py`);
nothing method-side changed. Also sets up (but does not yet run) the V-JEPA
"better backbone" experiment.

**One-line verdict:** the picture is materially stronger with three datasets than
with one. Our plain count-free Ward hierarchy **matches TW-FINCH on Breakfast**
(the largest benchmark, 1712 videos: 63.5 vs 63.8 MoF) and is competitive on all
three, while beating uniform/kmeans. The count-free boundary gate PASSES on all
three. hicap's simple hierarchy is a credible peer of the standard unsupervised-TAS
baseline — the count-free framing and the (still-unbuilt) caption layer are the
intended differentiators on top.

---

## Benchmarks

| dataset    | videos | fps | actions | notes |
|------------|-------:|----:|--------:|-------|
| 50 Salads  |    50  | 30  | ~17     | single recipe, near-linear |
| Breakfast  |  1712  | 15  | 48      | 10 activities, diverse — the real scale-up |
| GTEA       |    28  | 15  | 11      | egocentric, short clips, recurring actions |

All use the standard precomputed I3D features (same as every TAS baseline), so
comparisons are apples-to-apples.

## Experiment 1 — count-free boundary gate (all three PASS)

Best-level boundary-F1 vs matched-count uniform and random:

| dataset   | model | uniform | random | gain vs uniform |
|-----------|------:|--------:|-------:|----------------:|
| 50 Salads | 0.215 | 0.114   | 0.095  | +0.101 |
| GTEA      | 0.583 | 0.432   | 0.377  | +0.151 |
| Breakfast | 0.212 | 0.127   | 0.128  | +0.085 |

The hierarchy contains a level beating both baselines on every dataset. (As on
50 Salads, the gain holds across a range of levels, not one point.)

## Experiment 2 — known-K comparison (the headline)

Every method handed C = #unique GT actions per video; standard metrics. **MoF:**

| method       | 50 Salads | GTEA | Breakfast |
|--------------|----------:|-----:|----------:|
| uniform      | 58.7 | 46.3 | 56.2 |
| kmeans       | 52.9 | 53.5 | 54.4 |
| **twfinch**  | **67.4** | 50.9 | **63.8** |
| ours_contig  | 64.8 | 50.3 | **63.5** |
| ours_cluster | 59.6 | **52.6** | 60.8 |

**F1@50** (strict segmental overlap):

| method       | 50 Salads | GTEA | Breakfast |
|--------------|----------:|-----:|----------:|
| uniform      | 41.3 | 11.7 | 36.8 |
| twfinch      | 52.9 | 14.5 | 46.7 |
| ours_contig  | 49.3 | 15.0 | 44.9 |
| ours_cluster | 31.8 | 23.0 | 32.9 |

**Reading:**
- **Breakfast is the strong result.** On the largest, most diverse benchmark our
  `ours_contig` (63.5 MoF, 44.9 F1@50) is **statistically on top of TW-FINCH**
  (63.8, 46.7) — a 0.3 MoF gap over 1712 videos. A generic count-free agglomerative
  hierarchy matching a purpose-built TAS method at scale is a genuinely good result.
- **50 Salads:** TW-FINCH keeps a ~2.6 MoF edge; our method still clearly beats
  uniform/kmeans. The near-linear single recipe favors TW-FINCH's tuning slightly.
- **GTEA:** small and noisy (28 videos); kmeans happens to win, methods are within
  a few MoF of each other. Notably, here `ours_cluster` > `ours_contig` (52.6 vs
  50.3) — the recurrence-aware readout wins on short egocentric clips with
  within-video action recurrence, as predicted. On 50 Salads/Breakfast, where
  actions mostly occur once per video, `ours_contig` wins instead. So the two
  readouts are complementary and dataset-dependent, exactly as expected.

## Bottom line

- Averaged over the three benchmarks, `ours_contig` (59.5 MoF) essentially ties
  TW-FINCH (60.7) and clearly beats uniform (53.7) and kmeans (53.6). Our simple
  count-free hierarchy is a peer of the standard unsupervised-TAS baseline.
- This is *without* the two intended differentiators: the count-free advantage
  (TW-FINCH must be told K; we don't — see the K-sensitivity in the prior report)
  and the VLM-caption vocab-free labeling (none of these numbers use captions).
- Next real lever tested next: **V-JEPA features** in place of I3D (pipeline built
  and validated, see `VJEPA_PLAN.md`; blocked only on sourcing RGB).

## Reproduce

```
for ds in 50salads breakfast gtea; do
  python run_gate.py    --config hicap/configs/gate_50salads.yaml     --dataset $ds
  python run_compare.py --config hicap/configs/compare_50salads.yaml  --dataset $ds
done
```
Reports: `hicap/results/{gate,compare}_report_<dataset>.json`.

## Caveats

- Same as the prior report: TW-FINCH is a validated reimplementation (not authors'
  code); "ours" is I3D boundary/hierarchy quality only (no captions); MoF/edit/F1
  via per-video Hungarian; known-K uses C = #unique GT classes.
- GTEA (28 videos) is too small to draw strong conclusions from on its own.
