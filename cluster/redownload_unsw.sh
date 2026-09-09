#!/bin/bash
# Re-download UNSW both files from one repo (srsds/netattack) for consistent schema.
set -euo pipefail
cd ~/mgnn/data
BASE="https://raw.githubusercontent.com/srsds/netattack/master"
echo "Downloading training-set (single source)..."
curl -sSL --max-time 600 -o UNSW_NB15_training-set.csv "$BASE/UNSW_NB15_training-set.csv"
echo "Downloading testing-set (single source)..."
curl -sSL --max-time 600 -o UNSW_NB15_testing-set.csv "$BASE/UNSW_NB15_testing-set.csv"
echo "--- headers training ---"
head -1 UNSW_NB15_training-set.csv
echo "--- headers testing ---"
head -1 UNSW_NB15_testing-set.csv
ls -la UNSW_NB15_*.csv