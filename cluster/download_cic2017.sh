#!/bin/bash
# Download CIC-IDS2017 all per-day CSVs from official mirror, then preprocess.
set -euo pipefail
cd ~/mgnn
source .venv/bin/activate
export PYTHONPATH="$PWD"
mkdir -p data logs

BASE="https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSV/MachineLearningCVE/"
FILES=(
"Monday-WorkingHours.pcap_ISCX.csv"
"Tuesday-WorkingHours.pcap_ISCX.csv"
"Wednesday-workingHours.pcap_ISCX.csv"
"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
"Friday-WorkingHours-Morning.pcap_ISCX.csv"
"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
cd data
for f in "${FILES[@]}"; do
  if [ -s "$f" ]; then echo "  $f exists, skip"; continue; fi
  echo "  downloading $f ..."
  curl -sSL --max-time 1200 -o "$f" "$BASE$f" || echo "  FAILED $f"
  echo "    $(du -h "$f" | cut -f1)"
done
echo "=== all CIC files done ==="
ls -la CIC*.csv *.pcap_ISCX.csv 2>/dev/null | awk '{print $5, $9}'
cd ..

# start preprocess in background after downloads
nohup python - <<'PY' > logs/cic2017_preprocess.log 2>&1 &
import time
from src.data.dataset import prepare_cic_ids2017
t0 = time.time()
data = prepare_cic_ids2017("data", download=False)
print("DONE cic_ids2017 in %.0fs" % (time.time()-t0), flush=True)
print("N=%d anom=%.4f feat=%s" % (data['n_total'], data['anomaly_rate'], data['features'].shape))
PY
echo "CIC2017 preprocess started, pid $!"