#!/usr/bin/env bash
# Download and preprocess all three benchmark datasets.
# Missing raw CSVs can be dropped into data/ and this script will pick them up.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONPATH="$PWD"
python - <<'PY'
from src.data.dataset import prepare_cic_ids2017, prepare_unsw_nb15, prepare_cse_ids2018

# n_samples=None -> full labeled set. Pass a cap only for a fast smoke test.
prepare_cic_ids2017("data", download=True,  n_samples=None)
prepare_unsw_nb15("data",   download=True,  n_samples=None)
prepare_cse_ids2018("data", download=True,  n_samples=None)
PY

echo "[download_prep] all datasets preprocessed under data/*.pt"