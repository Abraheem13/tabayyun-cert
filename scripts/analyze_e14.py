"""Score E14 (real multi-source tier, Flight data) against Amendment 19 (F1-F5).

Usage: python scripts/analyze_e14.py results/e14/e14_confirm.json
"""
import json
import sys
from pathlib import Path

import numpy as np

p = Path(sys.argv[1])
rows = json.loads(p.read_text())
FS = sorted({r["f"] for r in rows})
ATT = ["assert", "edge", "unbounded"]
SHOW = ["vote", "titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "robustrag", "smsr",
        "tma_corr", "tabayyun", "tabayyun_strict", "tabayyun_w"]
rng = np.random.default_rng(0)


def sel(f, d, a=None):
    return [r for r in rows if r["f"] == f and r["defence"] == d and (a is None or r["attack"] == a)]


def mean(rr, k):
    return float(np.mean([r[k] for r in rr])) if rr else float("nan")


def worst_attack(f, d):
    return max(ATT, key=lambda a: mean(sel(f, d, a), "installed"))


def paired_ci(f, a, d1, d2, k="installed", B=4000):
    """95% bootstrap CI of mean(d1 - d2), paired by (day, trial)."""
    A = {(r["day"], r["trial"]): r[k] for r in sel(f, d1, a)}
    Bm = {(r["day"], r["trial"]): r[k] for r in sel(f, d2, a)}
    keys = sorted(set(A) & set(Bm))
    diff = np.array([A[x] - Bm[x] for x in keys])
    bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return float(diff.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(diff)


out = {"table": {}, "verdict": {}}
for f in FS:
    out["table"][f] = {}
    for d in SHOW:
        cl = sel(f, d, "none")
        if not cl:
            continue
        e = {"clean_accuracy": round(mean(cl, "recall"), 3),
             "worst_installed": round(max(mean(sel(f, d, a), "installed") for a in ATT), 3),
             "worst_amnesia": round(max(mean(sel(f, d, a), "amnesia") for a in ATT), 3)}
        if "certified_correct_frac" in cl[0]:
            e["certified_correct_clean"] = round(mean(cl, "certified_correct_frac"), 3)
            e["worst_served_installed"] = round(max(mean(sel(f, d, a), "served_installed") for a in ATT), 3)
            e["violations"] = int(sum(r["violations"] + r["action_violations"] for r in sel(f, d)))
        out["table"][f][d] = e
T = out["table"]
viol = sum(e.get("violations", 0) for f in FS for e in T[f].values())
f2 = [(f, d, T[f][d]["worst_installed"]) for f in FS for d in ("tabayyun_strict", "tabayyun_w")
      if T[f][d]["worst_installed"] > 0.05]
f3 = {f: (T[f]["tabayyun_w"]["clean_accuracy"], T[f]["vote"]["clean_accuracy"]) for f in FS}
f4 = {f: T[f]["tabayyun_w"]["certified_correct_clean"] for f in FS}
f5 = {}
for f in FS:
    if f < 2:
        continue
    a = worst_attack(f, "agg_trmean")
    f5[f] = {"attack": a, "trmean_minus_w": [round(x, 3) if isinstance(x, float) else x
                                              for x in paired_ci(f, a, "agg_trmean", "tabayyun_w")]}
out["F1_violations"] = viol
out["F2_exceptions"] = f2
out["F3_W_vs_vote_accuracy"] = f3
out["F4_W_certified_correct"] = f4
out["F5_trmean_minus_W_installed_CI"] = f5
out["verdict"] = {"F1": viol == 0, "F2": not f2,
                  "F3": all(w >= v - 0.03 for w, v in f3.values()),
                  "F4": all(x >= 0.80 for x in f4.values()),
                  "F5": all(v["trmean_minus_w"][1] > 0 for v in f5.values())}
print(json.dumps(out, indent=1))
p.with_name(p.stem + "_scorecard.json").write_text(json.dumps(out, indent=1))
