"""Clean-only choice of the agreement tolerance eps of Tabayyun-W on Tier B (MemLM).

Looks ONLY at attack = none, exploratory trials 0-4, models d128 and d256, all 16 cells.
Objective (as for the other Tier-B variants): clean recall + certified-and-correct fraction.
Rule fixed in advance (security first): the SMALLEST eps on the grid within 0.02 of the best.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e04_mqar_attack as E  # noqa: E402

GRID = (0.1, 0.2, 0.3, 0.4, 0.6)


def main():
    torch.set_grad_enabled(False)
    cells = [(rg, n, f, rho, R) for rg, n, fs in [("dense", 10, (1, 2)), ("locality", 20, (2, 3))]
             for f in fs for rho in (0.4, 0.7) for R in (1, 3)]
    out = {}
    for eps in GRID:
        E.EPS_W = eps
        rec, cc = [], []
        for tag in ("d128", "d256"):
            model = E.load(ROOT / "checkpoints" / f"memlm_{tag}.pt", "cpu")
            for (rg, n, f, rho, R) in cells:
                for s in range(5):
                    r = E.run_trial(model, model.cfg.vocab, 16, n, f, rho, rg, "none", s, "cpu", ["tabayyun_w"], R)["tabayyun_w"]
                    rec.append(r["recall"]); cc.append(r["certified_correct_frac"])
        out[str(eps)] = {"clean_recall": float(np.nanmean(rec)), "certified_correct": float(np.mean(cc)),
                         "objective": float(np.nanmean(rec) + np.mean(cc))}
        print(eps, out[str(eps)], flush=True)
    best = max(v["objective"] for v in out.values())
    out["selected_eps"] = min(e for e in GRID if out[str(e)]["objective"] >= best - 0.02)
    print("selected eps", out["selected_eps"])
    (ROOT / "results" / "e05").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "e05" / "tuning_w_clean.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
