"""Clean-only choice of the memory write step theta for Tier C (E08).

Looks ONLY at attack = none, exploratory trials 0-4.  Objective (fixed before
looking): mean over cells of Tabayyun's clean opt_frac + certified fraction;
the same theta is then used for every defence.  Baseline clean opt_frac is
reported for transparency.
"""
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e08_oran as e08  # noqa: E402  (importable by name, so spawn workers on Windows can unpickle it)

THETAS = [0.5, 0.2, 0.1, 0.05]


def main():
    defs = ["tabayyun_anchored2", "titans", "agg_trmean", "agg_median", "robustrag", "tma_corr"]
    jobs = [(rg, n, f, rho, "none", d, s, 10, th)
            for th, (rg, n, f), rho, d, s in itertools.product(THETAS, e08.CELLS, (0.4, 0.7), defs, range(5))]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        rows = list(ex.map(e08.one, jobs, chunksize=2))
    for j, r in zip(jobs, rows):
        r["theta"] = j[8]
    out = {}
    for th in THETAS:
        e = {}
        for d in defs:
            rr = [r for r in rows if r["theta"] == th and r["defence"] == d]
            e[d] = {"opt_frac": round(float(np.mean([r["opt_frac"] for r in rr])), 3)}
            if d.startswith("tab"):
                e[d]["certified"] = round(float(np.mean([r["certified"] for r in rr])), 3)
                e[d]["objective"] = round(e[d]["opt_frac"] + e[d]["certified"], 3)
        out[str(th)] = e
        print(th, json.dumps(e))
    best = max(THETAS, key=lambda t: out[str(t)]["tabayyun_anchored2"]["objective"])
    out["selected_theta"] = best
    print("selected theta", best)
    (ROOT / "results" / "e08").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e08" / "tuning_clean.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
