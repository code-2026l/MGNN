#!/bin/bash
# Preprocess CIC-IDS2017 (8 daily CSVs) -> cic_ids2017.pt
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
nohup python - <<'PY' > logs/cic2017_preprocess.log 2>&1 &
import time
from src.data.dataset import prepare_cic_ids2017
t0 = time.time()
data = prepare_cic_ids2017("data", download=False)
print("DONE cic_ids2017 in %.0fs" % (time.time()-t0), flush=True)
print("N=%d anom=%.4f feat=%s" % (data['n_total'], data['anomaly_rate'], data['features'].shape))
PY
echo "CIC2017 preprocess started, pid $!"