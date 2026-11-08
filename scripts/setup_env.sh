#!/usr/bin/env bash
# One-time setup on the RTX 5080 machine (Linux / WSL2).
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python scripts/check_gpu.py | tee results/env.txt
pytest -q
