"""Score Amendment 16 (Tabayyun-W, static protocol on Tiers B and D, union attacker on Tier D).

Usage: python scripts/analyze_e16.py --static results/e04/e04_d128_w_confirm.json ... --union results/e11/e11_gpt2xl_w_union_summary.json ...
Each --static file must already have its _scorecard.json (scripts/analyze_e04.py) and _summary.json.
"""
import json
import sys
from pathlib import Path

args = sys.argv[1:]
static = args[args.index("--static") + 1: args.index("--union")] if "--union" in args else args[args.index("--static") + 1:]
union = args[args.index("--union") + 1:] if "--union" in args else []
out, ok = {}, {"H16a": True, "H16b": True, "H16c": True, "H16d": True, "H16e": True, "H16f": True}
for p in static:
    sc = json.loads(Path(p.replace(".json", "_scorecard.json")).read_text())
    summ = json.loads(Path(p.replace(".json", "_summary.json")).read_text())
    cells = sc["cells"]
    W = {c: v["tabayyun_w"] for c, v in cells.items()}
    viol = sum(w["H3_violations"] for w in W.values())
    p1 = sum(w["P1_pass"] for w in W.values()); c1 = sum(w["C1_pass"] for w in W.values())
    h7 = sum(w["H7_pass"] for w in W.values())
    loc = [c for c in cells if c.startswith("locality")]
    rw = sum(cells[c]["tabayyun_w"]["table"]["tabayyun_w"]["rec"] for c in loc) / len(loc)
    rs = sum(cells[c]["tabayyun_strict"]["table"]["tabayyun_strict"]["rec"] for c in loc) / len(loc)
    c2 = sum(cells[c]["tabayyun_w"].get("C2_pass", False) for c in loc)
    served = max(v["worst_served_installed"] for k, v in summ.items() if k.endswith("|tabayyun_w"))
    n = len(cells)
    out[Path(p).name] = {"violations": viol, "P1": f"{p1}/{n}", "C1": f"{c1}/{n}", "C1_dominated_by":
                         {c: w["C1_dominated_by"] for c, w in W.items() if not w["C1_pass"]},
                         "H7_f+1": f"{h7}/{n}", "locality_recall_W": round(rw, 3), "locality_recall_strict": round(rs, 3),
                         "C2_descriptive": f"{c2}/{len(loc)}", "worst_installed_W": max(w["table"]["tabayyun_w"]["inst"] for w in W.values()),
                         "worst_served_installed_W": served}
    ok["H16a"] &= viol == 0
    ok["H16b"] &= p1 == n
    ok["H16c"] &= c1 == n
    ok["H16d"] &= h7 == n
    ok["H16e"] &= rw >= rs + 0.10
for p in union:
    S = json.loads(Path(p).read_text())
    cells = sorted({k.rsplit("|", 1)[0] for k in S})
    w = [S[f"{c}|tabayyun_w"] for c in cells]
    r = {"violations": sum(x.get("violations", 0) for x in w), "worst_installed": max(x["pgd_installed"] for x in w),
         "worst_served_installed": max(x["served_installed"] for x in w),
         "union_vs_static_mean": (round(sum(x["pgd_installed"] for x in w) / len(w), 4),
                                  round(sum(x["static_installed"] for x in w) / len(w), 4)),
         "amnesia_mean": round(sum(x["amnesia_mean"] for x in w) / len(w), 3)}
    out[Path(p).name] = r
    ok["H16a"] &= r["violations"] == 0
    ok["H16f"] &= r["worst_installed"] <= 0.05 and r["worst_served_installed"] <= 0.02
out["verdict"] = ok
print(json.dumps(out, indent=1))
Path("results/scorecard_amend16.json").write_text(json.dumps(out, indent=1))
