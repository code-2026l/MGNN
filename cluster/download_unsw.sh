#!/bin/bash
# Download UNSW-NB15 labeled CSV and preprocess to unsw_nb15.pt
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
mkdir -p data

cd data
echo "Downloading UNSW_NB15_training-set.csv ..."
curl -sSL --max-time 600 -o UNSW_NB15_training-set.csv "https://raw.githubusercontent.com/CharlesMure/cassiope-NIDS/master/Data/UNSW-NB15/UNSW_NB15_training-set.csv"
echo "Downloading UNSW_NB15_testing-set.csv ..."
curl -sSL --max-time 600 -o UNSW_NB15_testing-set.csv "https://raw.githubusercontent.com/srsds/netattack/master/UNSW_NB15_testing-set.csv"
ls -la UNSW_NB15_*.csv

cd ..
echo "Preprocessing UNSW-NB15 ..."
nohup python - <<'PY' > logs/unsw_preprocess.log 2>&1 &
import time
from src.data.dataset import prepare_unsw_nb15
t0 = time.time()
data = prepare_unsw_nb15("data")
print("DONE unsw_nb15 in %.0fs" % (time.time()-t0), flush=True)
print("N=%d anom=%.4f feat=%s" % (data['n_total'], data['anomaly_rate'], data['features'].shape))
PY
echo "UNSW preprocess started in background, pid $!"