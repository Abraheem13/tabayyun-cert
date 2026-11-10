"""Clean-only tuning of the Tier-B gate (Amendment 4 rule: attack = none, trials 0-99 only).

Sweeps nu_rel x gamma for both models and writes results/e04/tuning_clean.json.
Objective: maximise mean over cells of (clean recall + certified-AND-correct fraction).
(Changed before any tuning data was produced: the Day-8 PI run crashed on a script bug,
and a timing check showed that "certified fraction" alone rewards empty memories, whose
empty read trivially equals the empty reference.)
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
out = {}
for model, nu, gamma in itertools.product(["d128", "d256"], ["none", "0.05", "0.1", "0.2", "0.3", "0.5"],
                                         ["0.0", "0.3", "0.5"]):
    tag = f"tune_{model}_nu{nu}_g{gamma}"
    cmd = [PY, "experiments/e04_mqar_attack.py", "--device", DEVICE, "--ckpt", f"checkpoints/memlm_{model}.pt", "--trials", "100",
           "--trial_offset", "0", "--attacks", "none", "--defences", "tabayyun", "--gamma", gamma, "--tag", tag]
    if nu != "none":
        cmd += ["--nu_rel", nu]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    summ = json.loads((ROOT / "results" / "e04" / f"e04_{tag}_summary.json").read_text())
    rec = [v["clean_recall"] for v in summ.values()]
    cert = [v["certified_correct_clean"] for v in summ.values()]
    out[tag] = {"model": model, "nu_rel": nu, "gamma": gamma, "mean_recall": sum(rec) / len(rec),
                "mean_certified": sum(cert) / len(cert), "objective": sum(rec) / len(rec) + sum(cert) / len(cert),
                "cells": summ}
    print(f"{tag:28} recall={out[tag]['mean_recall']:.3f} certified={out[tag]['mean_certified']:.3f}", flush=True)
    for p in (ROOT / "results" / "e04").glob(f"e04_{tag}*.json"):
        p.unlink()
for model in ["d128", "d256"]:
    best = max((v for v in out.values() if v["model"] == model), key=lambda v: v["objective"])
    print(f"BEST {model}: nu_rel={best['nu_rel']} gamma={best['gamma']} objective={best['objective']:.3f}")
(ROOT / "results" / "e04" / "tuning_clean.json").write_text(json.dumps(out, indent=1))
