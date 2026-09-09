#!/bin/bash
# Runs Table 6 paper-condition reproduction on real data with worker-parallel
# loading so the A100 is actually used (previous run starved the GPU at ~11%).
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
mkdir -p logs
rm -f logs/table6_paper.log
nohup python -u experiments/run_table6.py \
  --data-dir data --device cuda:0 \
  --runs 5 --epochs 30 --batch-size 2048 --hidden 128 \
  --paper --datasets cic_ids2017,unsw_nb15 \
  > logs/table6_paper.log 2>&1 &
echo "table6 paper launcher started, pid $!"