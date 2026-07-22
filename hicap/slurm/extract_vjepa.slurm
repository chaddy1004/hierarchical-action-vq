#!/bin/bash
#SBATCH --job-name=hicap-vjepa
#SBATCH --account=def-swasland-ab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# V-JEPA feature extraction on Rorqual. Compute nodes have NO internet, so RGB and
# the V-JEPA weights must already be on scratch before submitting (weights are at
# /home/chaddy/scratch/models/vjepa2-vitg-fpc64-384; download RGB on a login node).
#
# Submit:  sbatch hicap/slurm/extract_vjepa.slurm hicap/configs/vjepa_breakfast.yaml
# Resumable: re-submitting skips already-extracted videos (idempotent).

set -euo pipefail
CONFIG="${1:-hicap/configs/vjepa_breakfast.yaml}"

module load StdEnv/2023 python/3.11 cuda
source ~/havq-env/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

cd ~/hierarchical-action-vq
python run_vjepa_extract.py --config "$CONFIG"
