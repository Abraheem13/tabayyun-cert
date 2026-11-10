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
AMEND2_DEFS = ("titans,titans_clip,miras_yaad,agg_mean,agg_trmean,agg_median,agg_krum,agg_geomed,agg_cclip,"
               "agg_median_anc,agg_krum_anc,smsr,robustrag,tma_corr,tabayyun_anchored2")
AMEND2_BASE = ("agg_mean,agg_trmean,agg_median,agg_krum,agg_geomed,agg_cclip,agg_median_anc,agg_krum_anc,"
               "smsr,robustrag,tma_corr,miras_yaad,titans_clip")
AMEND1_DEFS = ("titans,titans_clip,titans_ttl,agg_mean,agg_trmean,agg_median,agg_krum,agg_geomed,agg_cclip,"
               "agg_median_anc,agg_trmean_anc,agg_krum_anc,tab_nocorr,tabayyun,tabayyun_anchored,"
               "tabayyun_anchored2,tabayyun_anchored2_add")


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
        # Amendment 1: confirmatory test of anchored2 + H7 + fairness (A11) + A3 on fresh seeds 110-119
        5: [[PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "110",
             "--n", "10", "--support", "all", "--tag", "dense_amend1", "--defences", AMEND1_DEFS],
            [PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "110",
             "--n", "20", "--support", "minority", "--fracs", "0.1,0.15", "--tag", "locality_amend1",
             "--defences", AMEND1_DEFS],
            [PY, "scripts/analyze_e02.py", "--method", "tabayyun_anchored2", "--out", "scorecard_amend1",
             "results/e02/e02_dense_amend1.json", "results/e02/e02_locality_amend1.json"],
            [PY, "scripts/analyze_e02.py", "--method", "tabayyun_anchored", "--out", "scorecard_day4",
             "results/e02/e02_dense.json", "results/e02/e02_locality.json"]],
        # Amendment 2: 2026 memory defences added to the comparison set, fresh seeds 120-129
        6: [[PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "120",
             "--n", "10", "--support", "all", "--tag", "dense_amend2", "--defences", AMEND2_DEFS],
            [PY, "experiments/e02_synthetic_defence.py", "--seeds", "10", "--seed_offset", "120",
             "--n", "20", "--support", "minority", "--fracs", "0.1,0.15", "--tag", "locality_amend2",
             "--defences", AMEND2_DEFS],
            [PY, "scripts/analyze_e02.py", "--method", "tabayyun_anchored2", "--out", "scorecard_amend2",
             "--baselines", AMEND2_BASE,
             "results/e02/e02_dense_amend2.json", "results/e02/e02_locality_amend2.json"]],
        # Tier B: train two memory LMs on GPU, then exploratory attack runs (trials 0-99)
        7: [[PY, "experiments/e03_mqar_train.py", "--device", dev, "--steps", "8000", "--batch", "128",
             "--d_model", "128", "--dk", "64", "--max_pairs", "24", "--max_segments", "6", "--tag", "memlm_d128"],
            [PY, "experiments/e03_mqar_train.py", "--device", dev, "--steps", "8000", "--batch", "128",
             "--d_model", "256", "--dk", "128", "--max_pairs", "24", "--max_segments", "6", "--tag", "memlm_d256"],
            [PY, "experiments/e04_mqar_attack.py", "--ckpt", "checkpoints/memlm_d128.pt", "--device", dev,
             "--trials", "100", "--trial_offset", "0", "--tag", "d128_explore"],
            [PY, "experiments/e04_mqar_attack.py", "--ckpt", "checkpoints/memlm_d256.pt", "--device", dev,
             "--trials", "100", "--trial_offset", "0", "--tag", "d256_explore"]],
        # clean-only tuning of nu_rel / gamma (no attacked runs are looked at)
        8: [[PY, "scripts/tune_e04_clean.py"]],
        # Amendment 4: Tier-B confirmatory, trials 200-299, both models, R in {1,3}
        9: [[PY, "experiments/e04_mqar_attack.py", "--ckpt", "checkpoints/memlm_d128.pt", "--device", dev,
             "--trials", "100", "--trial_offset", "200", "--obs_rounds", "1,3", "--tag", "d128_confirm"],
            [PY, "scripts/analyze_e04.py", "results/e04/e04_d128_confirm.json"],
            [PY, "experiments/e04_mqar_attack.py", "--ckpt", "checkpoints/memlm_d256.pt", "--device", dev,
             "--trials", "100", "--trial_offset", "200", "--obs_rounds", "1,3", "--tag", "d256_confirm"],
            [PY, "scripts/analyze_e04.py", "results/e04/e04_d256_confirm.json"]],
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
