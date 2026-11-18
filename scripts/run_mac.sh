#!/usr/bin/env bash
# Day 24, E15 part (Amendment 20 + 21) on an Apple-silicon Mac (MPS).  E14 runs in the cloud container.
# Usage (from the repo root):  bash scripts/run_mac.sh
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install torch
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install -e .
fi
PY=.venv/bin/python
DEV=$($PY -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')")
mkdir -p results/day24
LOG=results/day24/console_mac.txt
echo "# Day 24 (Mac, device=$DEV) started $(date)" | tee "$LOG"
for spec in "gpt2-xl gpt2xl" "Qwen/Qwen2.5-1.5B qwen15b"; do
  set -- $spec
  $PY experiments/e15_midlayer.py --model "$1" --device "$DEV" --tune_w 2>&1 | tee -a "$LOG"
  $PY experiments/e15_midlayer.py --model "$1" --device "$DEV" --trials 10 --trial_offset 1500 --tag "$2_confirm" 2>&1 | tee -a "$LOG"
  git add results && git commit -qm "Day 24 E15 $2 results (Mac)" && git push -q   # push after each model
done
$PY scripts/analyze_e15m.py results/e15m/e15m_gpt2xl_confirm_summary.json results/e15m/e15m_qwen15b_confirm_summary.json 2>&1 | tee -a "$LOG"
echo "# finished $(date)" | tee -a "$LOG"
git add results && git commit -qm "Day 24 E15 results (Mac)" ; git push
