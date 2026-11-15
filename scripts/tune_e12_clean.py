"""Clean-only choice of the agreement tolerance eps of Tabayyun-W (Tier A).

Looks ONLY at attack = none, exploratory seeds 0-4.  Rule fixed in advance (security first):
the SMALLEST eps on the grid whose clean recall at m >= f+1 is within 0.02 of the best.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e12_witness as e12  # noqa: E402

GRID = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def main():
    jobs = [(n, f, "tabayyun_w", "none", s, eps) for eps in GRID for (n, f) in [(10, 2), (40, 2), (40, 3)]
            for s in range(5)]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(e12.one, jobs))
    out = {}
    for eps in GRID:
        rr = [r for j, rs in zip(jobs, res) if j[5] == eps for r in rs]
        out[str(eps)] = {"clean_recall_m_ge_f+1": float(np.mean([r["recall"] for r in rr if r["m"] >= r["f"] + 1])),
                         "clean_recall_m_le_f": float(np.mean([r["recall"] for r in rr if r["m"] <= r["f"]])),
                         "violations": int(sum(not r["cert_sound"] for r in rr))}
        print(eps, out[str(eps)])
    best = max(v["clean_recall_m_ge_f+1"] for v in out.values())
    out["selected_eps"] = min(e for e in GRID if out[str(e)]["clean_recall_m_ge_f+1"] >= best - 0.02)
    print("selected eps", out["selected_eps"])
    (ROOT / "results" / "e12").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e12" / "tuning_clean.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
