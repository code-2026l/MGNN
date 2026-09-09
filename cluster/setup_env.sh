#!/usr/bin/env bash
# One-time environment setup on the compute host (login node).
# Creates a venv and installs the pinned requirements (incl. PyTorch Geometric).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
echo "[setup_env] using interpreter: $($PY --version)"

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi
source .venv/bin/activate

# Install PyTorch for the target accelerator first (adjust the index if the
# cluster uses a custom wheel mirror), then the rest.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric
pip install -r requirements.txt

echo "[setup_env] done. Activate with: source .venv/bin/activate"