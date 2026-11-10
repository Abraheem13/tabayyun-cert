"""Cross-platform day runner (Windows PowerShell, Linux, macOS).

Usage:
    python scripts/run_day.py 1            # device defaults to cuda if available
    python scripts/run_day.py 2 --device cpu

Every command's output is saved to results/dayNN/console.txt (committed),
so the run can be reviewed from GitHub.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def plan(day: int, dev: str) -> list[list[str]]:
    days = {
        1: [[PY, "scripts/check_gpu.py"],
            [PY, "experiments/e01_persistence.py", "--smoke", "--device", dev],
            [PY, "experiments/e02_synthetic_defence.py", "--smoke", "--device", dev]],
        2: [[PY, "experiments/e01_persistence.py", "--device", dev, "--seeds", "5"]],
        3: [[PY, "experiments/e01_persistence.py", "--device", dev, "--part", "halflife",
             "--kinds", "mlp", "--streams", "random", "--tag", "mlp_random"],
            [PY, "experiments/e01_persistence.py", "--device", dev, "--part", "halflife",
             "--kinds", "linear", "--streams", "orthogonal", "--tag", "linear_orth"],
            [PY, "experiments/e01_persistence.py", "--device", dev, "--part", "rayn", "--seeds", "10"]],
        # E02 runs on CPU workers (tiny per-step ops are faster on CPU than on the GPU)
        4: [[PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "100",
             "--n", "10", "--support", "all", "--tag", "dense"],
            [PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "100",
             "--n", "20", "--support", "minority", "--fracs", "0.1,0.15", "--tag", "locality"]],
    }
    if day not in days:
        sys.exit(f"Day {day}: its commands are added to scripts/run_day.py on that day (see ROADMAP.md)")
    return [[PY, "-m", "pytest", "-q"]] + days[day]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day", type=int)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device
    if dev is None:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    out_dir = ROOT / "results" / f"day{a.day:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "console.txt"
    failed = []
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"# Day {a.day}  device={dev}  started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for cmd in plan(a.day, dev):
            line = " ".join(Path(c).name if c == PY else c for c in cmd)
            print(f"\n=== {line}", flush=True)
            fh.write(f"\n=== {line}\n")
            t = time.time()
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for out in p.stdout:
                print(out, end="", flush=True)
                fh.write(out)
            p.wait()
            msg = f"--- exit {p.returncode} in {time.time() - t:.1f}s\n"
            print(msg, end=""); fh.write(msg); fh.flush()
            if p.returncode != 0:
                failed.append(line)
        fh.write(f"\n# finished {time.strftime('%Y-%m-%d %H:%M:%S')}; failed: {failed or 'none'}\n")
    print(f"\nSaved log to {log.relative_to(ROOT)}")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
    print(f'\nNext:  git add results  ;  git commit -m "Day {a.day} results"  ;  git push')
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
