"""Post-hoc (Day 6) and prospective (Amendment 3) multi-objective analysis of E02.

For every cell: (1) Pareto check over (worst ASR, worst amnesia, worst installed, clean Recall_L2),
(2) best *secure* baseline (worst installed <= 0.1 and worst flip-ASR <= 0.2) on recall,
(3) best *useful* baseline (recall >= method - 0.05) on security.
Usage: python scripts/pareto_e02.py results/e02/scorecard_amend2.json [--method tabayyun_anchored2]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("scorecard")
ap.add_argument("--method", default="tabayyun_anchored2")
a = ap.parse_args()
d = json.load(open(a.scorecard))
M = a.method
out = {}
for reg, R in d.items():
    print(f"===== {reg}")
    out[reg] = {}
    for cell, c in R["cells"].items():
        t, inst = c["table"], c.get("installed_worst", {})
        def vec(x):
            return (t[x]["asr_worst"], t[x]["amnesia_worst"], inst.get(x, 0.0), -t[x]["clean_recall_l2"])
        m = vec(M)
        others = [x for x in t if x != M]
        dominators = [x for x in others if all(o <= mm + 1e-9 for o, mm in zip(vec(x), m)) and vec(x) != m]
        dominated = [x for x in others if all(mm <= o + 1e-9 for o, mm in zip(vec(x), m)) and vec(x) != m]
        secure = [x for x in others if inst.get(x, 1) <= 0.1 and t[x]["asr_worst"] <= 0.2]
        best_secure = max(secure, key=lambda x: t[x]["clean_recall_l2"]) if secure else None
        useful = [x for x in others if t[x]["clean_recall_l2"] >= t[M]["clean_recall_l2"] - 0.05]
        best_useful = min(useful, key=lambda x: (inst.get(x, 1), t[x]["asr_worst"])) if useful else None
        r = {"dominated_by": dominators, "dominates": dominated,
             "best_secure": best_secure,
             "recall_gap_vs_best_secure": t[M]["clean_recall_l2"] - (t[best_secure]["clean_recall_l2"] if best_secure else 0.0),
             "best_useful": best_useful,
             "installed_gap_vs_best_useful": (inst.get(M, 0) - inst.get(best_useful, 0)) if best_useful else None,
             "asr_gap_vs_best_useful": (t[M]["asr_worst"] - t[best_useful]["asr_worst"]) if best_useful else None}
        out[reg][cell] = r
        bs = f"{best_secure}({t[best_secure]['clean_recall_l2']:.2f})" if best_secure else "none"
        bu = (f"{best_useful}(inst={inst.get(best_useful, 0):.2f},ASR={t[best_useful]['asr_worst']:.2f})"
              if best_useful else "none")
        print(f"{cell:15} method: rec={t[M]['clean_recall_l2']:.2f} ASR={t[M]['asr_worst']:.2f} inst={inst.get(M, 0):.2f} | "
              f"dominated by: {dominators or 'NONE'} | dominates {len(dominated)}/{len(others)} | "
              f"best secure: {bs} | best useful: {bu}")
json.dump(out, open(a.scorecard.replace(".json", "_pareto.json"), "w"), indent=1)
