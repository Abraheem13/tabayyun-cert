#!/usr/bin/env bash
# Usage: scripts/run_day.sh <day> [device]
# Runs the experiments for a roadmap day, then prints what to commit.
set -euo pipefail
DAY=${1:?day}; DEV=${2:-cuda}
pytest -q
case "$DAY" in
  1) python scripts/check_gpu.py | tee results/env.txt
     python experiments/e01_persistence.py --smoke --device "$DEV"
     python experiments/e02_synthetic_defence.py --smoke --device "$DEV" ;;
  2) python experiments/e01_persistence.py --device "$DEV" --seeds 5 ;;
  4) python experiments/e02_synthetic_defence.py --device "$DEV" --seeds 10 --rounds 10 ;;
  *) echo "Day $DAY: command is added to this script on that day (see ROADMAP.md)"; exit 1 ;;
esac
echo; echo "Now: git add results && git commit -m \"Day $DAY results\" && git push"
