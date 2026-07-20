#!/bin/bash
#SBATCH --account=def-swasland-ab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16      # Rorqual H100 node = 64 cpu / 4 gpu -> 16 cpu/GPU (max for a 1-GPU job)
#SBATCH --mem=96G               # ~128G available per GPU; 96G is safe for up to ~8 packed processes
#SBATCH --time=12:00:00
#SBATCH --job-name=prep_c10
#SBATCH --chdir=/home/chaddy/hierarchical-action-vq
#SBATCH --output=JOBS/logs/prep_c10_%j.out

# Pack N extraction processes onto ONE H100. Extraction is batch=1 (one clip per
# forward), so a single process underuses the GPU; running several in parallel
# raises throughput until the 16 CPUs saturate on video decode.
#
# Usage:  sbatch JOBS/preprocess_clip10.slurm <shard> [<shard> ...]
#   sbatch JOBS/preprocess_clip10.slurm 0 1 2 3 4 5 6 7     # all 8 on one GPU (1 job)
#   sbatch JOBS/preprocess_clip10.slurm 0 1 2 3             # 4 on one GPU; run a
#   sbatch JOBS/preprocess_clip10.slurm 4 5 6 7             #   second job for the rest
# Idempotent (skip-if-cached): a timeout just needs a resubmit. Watch clips/sec
# in the log -- if an 8-pack matches a 4-pack, the CPUs are the bottleneck.

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
  python run_preprocessing.py \
    --config havq/data/preprocessing/configs/rorqual_vjepa_clip10.yaml \
    --shard-file "shards/shard_${s}.txt" &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "done (exit $rc): shards $*"
exit "$rc"
