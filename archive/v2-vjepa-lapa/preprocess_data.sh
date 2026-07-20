#!/usr/bin/env bash
# Preprocessing (PLAN.md Exp 1): cache V-JEPA features for HD-EPIC at multiple
# temporal scales. The two EXTREME scales run in parallel, one per GPU:
#   GPU 0 -> clip10 (1/3 s)  [longest job: most clips]
#   GPU 1 -> clip90 (3 s)    [shortest job: frees up first]
# The middle scale (clip30) is commented out below -- run it manually on
# whichever GPU frees first (clip90 on GPU 1 finishes first).
#
# Each job is idempotent (videos with an existing .npy are skipped), so a
# re-run resumes where it stopped. Per-job stdout goes to logs/.
set -euo pipefail

cd "$(dirname "$0")"

# V-JEPA loads from a local snapshot -- stay fully offline.
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_DIR="havq/data/preprocessing/configs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[GPU0] clip10 (1/3 s) -> $LOG_DIR/clip10.log"
CUDA_VISIBLE_DEVICES=0 uv run python run_preprocessing.py \
    --config "$CONFIG_DIR/vjepa_clip10.yaml" > "$LOG_DIR/clip10.log" 2>&1 &
PID10=$!

echo "[GPU1] clip90 (3 s)  -> $LOG_DIR/clip90.log"
CUDA_VISIBLE_DEVICES=1 uv run python run_preprocessing.py \
    --config "$CONFIG_DIR/vjepa_clip90.yaml" > "$LOG_DIR/clip90.log" 2>&1 &
PID90=$!

# clip30 (1 s) -- run manually once a GPU frees up (GPU 1 / clip90 finishes first):
#   CUDA_VISIBLE_DEVICES=1 uv run python run_preprocessing.py \
#       --config havq/data/preprocessing/configs/vjepa_clip30.yaml

wait $PID10 && echo "clip10 done" || echo "clip10 FAILED (see $LOG_DIR/clip10.log)"
wait $PID90 && echo "clip90 done" || echo "clip90 FAILED (see $LOG_DIR/clip90.log)"

echo "=== parallel preprocessing jobs finished ==="
