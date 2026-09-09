#!/bin/bash
# Preprocess UNSW-NB15 to unsw_nb15.pt (foreground, prints schema info).
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
mkdir -p data logs
python - <<'PY'
import time
from src.data.dataset import prepare_unsw_nb15
t0 = time.time()
data = prepare_unsw_nb15("data", download=False)
print("DONE unsw_nb15 in %.0fs" % (time.time()-t0), flush=True)
print("N=%d anom=%.4f feat=%s" % (data['n_total'], data['anomaly_rate'], data['features'].shape))
PY