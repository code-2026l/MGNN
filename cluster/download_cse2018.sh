#!/bin/bash
# Download and preprocess CSE-CIC-IDS2018 (all 10 daily files) in background.
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
mkdir -p data logs
nohup python - <<'PY' > logs/cse2018_download.log 2>&1 &
import time
from src.data.dataset import prepare_cse_ids2018
t0 = time.time()
data = prepare_cse_ids2018("data")
print("DONE cse_ids2018 in %.0fs" % (time.time()-t0), flush=True)
print("N=%d anom=%.4f feat=" % (data['n_total'], data['anomaly_rate']), data['features'].shape)
PY
echo "CSE2018 download started in background, pid $!"