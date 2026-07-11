#!/bin/bash
# Submit extraction job arrays. One array per feature scale; task i of each
# array processes shards/shard_i.txt (see make_shards.py / preprocess_rorqual.slurm).
# Safe to re-run: cached .npy files are skipped, so resubmitting after a
# timeout or to retry just one scale costs nothing extra.
#
# Usage:
#   ./cc_run_jobs.sh                 # submit all three scales
#   ./cc_run_jobs.sh clip90          # submit just one (e.g. to resubmit a straggler)
#   ./cc_run_jobs.sh clip10 clip30   # or a subset
set -euo pipefail
cd "$(dirname "$0")"
source ./cc_config.sh

SCALES=("$@")
if [ ${#SCALES[@]} -eq 0 ]; then
    SCALES=(clip10 clip30 clip90)
fi

mkdir -p logs
CFG_DIR="havq/data/preprocessing/configs"
MAX_IDX=$((HAVQ_N_SHARDS - 1))

for SCALE in "${SCALES[@]}"; do
    CFG="$CFG_DIR/rorqual_vjepa_${SCALE}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "Skipping $SCALE: $CFG not found (run ./cc_initialize.sh first)" >&2
        continue
    fi
    echo "Submitting havq-$SCALE: array 0-$MAX_IDX ($HAVQ_N_SHARDS shards)"
    sbatch --job-name="havq-$SCALE" \
           --array="0-$MAX_IDX" \
           --export=ALL,CFG="$CFG",N_SHARDS="$HAVQ_N_SHARDS" \
           preprocess_rorqual.slurm
done
