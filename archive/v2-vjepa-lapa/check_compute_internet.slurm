#!/bin/bash
#SBATCH --account=def-swasland-ab
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --job-name=net-test
#SBATCH --output=net-test-%j.out

echo "Testing outbound internet from compute node: $(hostname)"
curl -sI --max-time 10 https://huggingface.co && echo "huggingface.co: REACHABLE" || echo "huggingface.co: UNREACHABLE"
curl -sI --max-time 10 https://www.google.com && echo "google.com: REACHABLE" || echo "google.com: UNREACHABLE"
