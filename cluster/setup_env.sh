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

# PyTorch 2.1 (CUDA 12.1) as stated in the paper's implementation section.
pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric>=2.4.0
pip install -r requirements.txt

echo "[setup_env] done. Activate with: source .venv/bin/activate"