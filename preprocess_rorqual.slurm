#!/bin/bash
# Job array: task i extracts features for shards/shard_<i>.txt only. Shards are
# disjoint (see make_shards.py), so all array tasks can run concurrently with
# no race condition. If a task times out mid-shard, just resubmit the same
# array -- run_preprocessing.py skips videos that already have a cached .npy.
#
# Shard files are shared across feature scales (LPT balances by video
# duration, independent of clip length), so run make_shards.py ONCE and reuse
# shards/ for every CFG below.
#
# Submit one job array per feature scale, e.g.:
#   sbatch --export=ALL,CFG=havq/data/preprocessing/configs/rorqual_vjepa_clip10.yaml,N_SHARDS=8 \
#       --array=0-7 --job-name=havq-clip10 preprocess_rorqual.slurm
#
#SBATCH --account=def-swasland-ab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "$HOME/dev/hierarchical-action-vq"

module load StdEnv/2023 python/3.11 cuda
source ~/havq-env/bin/activate

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# N_SHARDS must match the --n-shards given to make_shards.py, and --array
# above must span 0-$((N_SHARDS-1)) -- this reconstructs the same zero-padded
# shard filename make_shards.py wrote for this array index.
MAX_IDX=$((N_SHARDS - 1))
N_DIGITS=${#MAX_IDX}
SHARD=$(printf "shards/shard_%0${N_DIGITS}d.txt" "$SLURM_ARRAY_TASK_ID")

echo "[task $SLURM_ARRAY_TASK_ID] config=$CFG shard=$SHARD"
python run_preprocessing.py --config "$CFG" --shard-file "$SHARD"
