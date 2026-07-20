#!/bin/bash
# Shared config for cc_initialize.sh and cc_run_jobs.sh -- edit the CHANGE_ME
# values below for your Rorqual filesystem layout, then:
#   1. run ./cc_initialize.sh once (sets up the venv, downloads weights,
#      writes the rorqual_vjepa_clip*.yaml configs, builds the shard files)
#   2. run ./cc_run_jobs.sh whenever you want to (re)submit extraction jobs

export HAVQ_ACCOUNT="def-swasland-ab"

# NOTE: ~/scratch (/home/chaddy/scratch) is NOT the real scratch filesystem --
# it's a plain directory that happens to sit inside the 50GB /home quota. The
# real 20TB scratch is at /lustre10/scratch/chaddy (symlinked as ~/links/scratch).
# Always use the resolved /lustre10/... path below, not ~/scratch or ~/links/scratch.

# Where the HD-EPIC videos live on Rorqual (already downloaded).
export HAVQ_VIDEOS_DIR="/lustre10/scratch/chaddy/datasets/HD-EPIC/Videos"

# Feature cache output. Regenerable, so it belongs on /scratch (not /project).
export HAVQ_FEATURES_DIR="/lustre10/scratch/chaddy/features/HD-EPIC"

# Backbone weights (see download_weights.py).
export HAVQ_WEIGHTS_DIR="/lustre10/scratch/chaddy/models"
export HAVQ_WEIGHT_MODELS="vjepa2-vitl-fpc64-256,vjepa2-vith-fpc64-256,vjepa2-vitg-fpc64-256,vjepa2-vitg-fpc64-384"
export HAVQ_DEFAULT_WEIGHT="vjepa2-vitg-fpc64-384"  # the one the generated configs actually point at

# Number of disjoint LPT-balanced shards == number of parallel GPU array tasks.
export HAVQ_N_SHARDS=8
