#!/bin/bash
#SBATCH --account=def-swasland-ab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=prep_c90
#SBATCH --chdir=/home/chaddy/hierarchical-action-vq
#SBATCH --output=JOBS/logs/prep_c90_%j.out

# clip90 (3 s) feature extraction -- ~1/9 the clips of clip10, so much faster.
# Same shards as clip10 (they're just video lists). Pack N processes onto one
# H100; idempotent (skip-if-cached), so a timeout/resubmit resumes.
#
# Usage:  sbatch JOBS/preprocess_clip90.slurm <shard> [<shard> ...]
#   sbatch JOBS/preprocess_clip90.slurm 0 1 2 3 4 5 6 7     # all 8 on one GPU

if [ "$#" -eq 0 ]; then
  echo "usage: sbatch $0 <shard_index> [<shard_index> ...]" >&2
  exit 1
fi

module load StdEnv/2023 python/3.11 cuda
source ~/havq-env/bin/activate
export HF_HUB_OFFLINE=1

echo "packing $# shard(s) onto one H100: $*"
pids=()
for s in "$@"; do
  python run_preprocessing.py --config havq/data/preprocessing/configs/rorqual_vjepa_clip90.yaml --shard-file "shards/shard_${s}.txt" &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "done (exit $rc): shards $*"
exit "$rc"
