#!/bin/bash
# Re-run paper-condition Table6 on CIC with unbuffered output for monitoring.
pkill -f 'experiments/run_table6.py' 2>/dev/null || true
sleep 2
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
nohup python experiments/run_table6.py \
  --data-dir data --device cuda:0 \
  --runs 5 --epochs 30 --batch-size 2048 --hidden 128 \
  --paper > logs/table6_cic_paper.log 2>&1 &
echo "restarted unbuffered, pid $!"