# V-JEPA "better backbone" experiment — readiness & plan

Goal: replace raw I3D features with V-JEPA 2 in the exact same hierarchy + eval
pipeline, to test whether a stronger backbone lifts the segmentation quality
(I3D + Ward was competitive-but-not-SOTA vs TW-FINCH, see `REPORT_2026-07-21.md`).
One-variable swap: only the features change.

## Status: pipeline built & locally validated; blocked only on RGB

**Ready:**
- Extraction code `hicap/data/vjepa_extract.py` (+ `run_vjepa_extract.py`) —
  sliding 64-frame window, stride-configurable, writes `(n_clips, 1408)` npy + meta
  in the same shape the pipeline reads. Reuses havq's proven V-JEPA extractor.
- Clip↔frame GT alignment `tas.load_labels_strided` — **tested**: downsamples
  per-frame GT to window centres so clip-resolution features score against the
  per-frame annotations. Verified boundaries land at the right clips.
- Extraction **runs** end-to-end (validated locally on a real video; model loads,
  emits 1408-d features).
- Rorqual is provisioned: `hicap` branch synced, `havq-env` has torch 2.13 +
  transformers + sklearn/scipy, V-JEPA weights already on scratch
  (`~/scratch/models/vjepa2-vitg-fpc64-384`), 19 TB scratch free. SLURM job
  `hicap/slurm/extract_vjepa.slurm` ready.

**Blocked:** the RGB videos for a TAS benchmark are not on disk anywhere.
- 50 Salads RGB: Dundee mirror dead.
- Breakfast RGB: the Brown/serre-lab hosting reorganized; the old
  `BreakfastII_15fps_qvga_sync.tar.gz` link is stale. RGB exists (public dataset,
  ~15 fps QVGA) but needs a current source.
- GTEA RGB: small; source not yet confirmed.
This is the one decision needed — see "What I need" below.

## The clip-length choice (the knob you flagged)

V-JEPA vitg-fpc64 pools a **64-frame window** into one vector. Two config knobs:
- `window` = 64 frames = 4.3 s @15 fps (Breakfast/GTEA) / 2.1 s @30 fps (50 Salads).
  How much time one embedding spans. 4.3 s is fine for coarse actions, a bit long
  for fine ones — can be reduced, though the model was trained at 64.
- `stride` = 8 frames (~2 tokens/s) in the Breakfast config → ~110 tokens for a
  900-frame video, a sensible resolution against ~10 annotated actions.

Both are in `hicap/configs/vjepa_breakfast.yaml`; tune stride for token rate,
window for temporal scale.

## Remaining code (small, deferred until features exist)

The gate/compare currently assume `len(features) == len(GT)` (per-frame I3D). For
clip-resolution V-JEPA they need a mode that: reads features from the V-JEPA dir,
reads GT via `load_labels_strided(stride, window, n_clips)` from the meta, and
converts `tol_sec` to clip-steps (`tol_sec * fps / stride`). ~10-line change to
each, best written against real features so it can be tested — not done blind.

## What I need to unblock

One of:
1. A working URL / source for Breakfast (or GTEA) RGB — I'll download it to
   `~/scratch/datasets/<ds>_rgb` on the rorqual login node and submit the SLURM job.
2. Or: point me at RGB you already have staged (locally or on the cluster).
3. Or: say "dig harder" and I'll keep searching for a live mirror.

Once RGB lands:
```
# on rorqual login node (has internet): download RGB to scratch, then
sbatch hicap/slurm/extract_vjepa.slurm hicap/configs/vjepa_breakfast.yaml
# then wire the clip-resolution eval mode and re-run gate/compare on V-JEPA feats
```

## Note on cost

Extraction speed is dominated by frame resolution. The local smoke test was slow
only because that video is 1408². Breakfast is QVGA (~320×240), so extraction of
1712 short clips on an H100 should be a few GPU-hours at most — well within budget.
