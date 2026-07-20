#!/bin/bash
# Runs a long-lived command protected against SSH disconnects, retrying on
# failure. Safe because both hd-epic-downloader.py and download_weights.py
# skip files that are already downloaded/verified, so a retry just resumes.
#
# Usage (run inside tmux so the command survives an SSH drop even without this
# script's own retry loop):
#   tmux new -s dl
#   bash run_protected.sh "python /home/chaddy/dev/hd-epic-downloader/hd-epic-downloader.py /scratch/chaddy/datasets --videos"
#   # Ctrl-b d to detach, then `tmux attach -t dl` later to check on it

set -u
CMD="$1"
MAX_RETRIES=20
LOG="protected_run_$(date +%Y%m%d_%H%M%S).log"

for i in $(seq 1 "$MAX_RETRIES"); do
    echo "$(date): attempt $i/$MAX_RETRIES: $CMD" | tee -a "$LOG"
    nice -n 19 bash -c "$CMD" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        echo "$(date): completed successfully." | tee -a "$LOG"
        exit 0
    fi
    echo "$(date): command failed, retrying in 30s..." | tee -a "$LOG"
    sleep 30
done

echo "$(date): gave up after $MAX_RETRIES attempts -- check $LOG" | tee -a "$LOG"
exit 1
