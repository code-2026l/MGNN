#!/bin/bash
# Correct LFS fetch: register smudge filter, then pull real objects.
set -euo pipefail
export PATH="$HOME/bin:$PATH"
cd ~/mgnn/data
rm -f *.pcap_ISCX.csv

# initialize LFS for this repo so git knows the smudge/clean filters
cd .cicrepo
git lfs install --local 2>&1 | tail -2 || git lfs install 2>&1 | tail -2
echo "=== pulling LFS objects ==="
time git lfs pull 2>&1 | tail -12
echo "=== sizes after pull ==="
ls -la *.csv 2>/dev/null | awk '{printf "%10.1fMB  %s\n", $5/1048576, $9}'
echo "=== sanity: first line of Monday ==="
head -c 200 Monday-WorkingHours.pcap_ISCX.csv 2>/dev/null; echo
cd ..
echo "=== copy to data dir ==="
cp .cicrepo/*.csv .
ls -la *.pcap_ISCX.csv | awk '{printf "%10.1fMB  %s\n", $5/1048576, $9}'
echo "CIC REAL DOWNLOAD DONE"