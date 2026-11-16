"""Score Amendment 15 (Tabayyun-W and certify-or-abstain under the union attacker, Tier B).

Usage: python scripts/analyze_e15.py results/e05/e05_d128_w_summary.json ... [--control results/e05/e05_d128_wcontrol_summary.json]
"""
import json
import sys
from pathlib import Path

args = sys.argv[1:]
control = None
if "--control" in args:
    i = args.index("--control"); control = args[i + 1]; args = args[:i] + args[i + 2:]
res = {}
ok = {"H15a": True, "H15b": True, "H15c": True, "H15d": True, "H15e": True}
for p in args:
    S = json.loads(Path(p).read_text())
    cells = sorted({k.rsplit("|", 1)[0] for k in S})
    g = lambda d, m: [S[f"{c}|{d}"][m] for c in cells]  # noqa: E731
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    viol = sum(sum(g(d, "violations")) for d in ("tabayyun_strict", "tabayyun_w"))
    w_inst = max(g("tabayyun_w", "pgd_installed"))
    served = {d: max(g(d, "served_installed")) for d in ("tabayyun_strict", "tabayyun_w")}
    amn = {d: round(mean(g(d, "amnesia_mean")), 3) for d in ("tabayyun_strict", "tabayyun_w")}
    sanity = {d: (round(mean(g(d, "pgd_installed")), 4), round(mean(g(d, "static_installed")), 4))
              for d in ("tabayyun_strict", "tabayyun_w", "agg_trmean")}
    r = {"violations": viol, "w_worst_installed": w_inst, "served_installed_worst": served, "amnesia_mean": amn,
         "union_vs_static_mean": sanity,
         "served_correct_mean": {d: round(mean(g(d, "served_correct")), 3) for d in ("tabayyun_strict", "tabayyun_w")},
         "worst_installed": {d: round(max(g(d, "pgd_installed")), 3) for d in ("tabayyun_strict", "tabayyun_w", "agg_trmean")}}
    ok["H15a"] &= viol == 0
    ok["H15b"] &= w_inst <= 0.05
    ok["H15c"] &= all(v <= 0.02 for v in served.values())
    ok["H15d"] &= amn["tabayyun_w"] <= amn["tabayyun_strict"] - 0.10
    ok["H15e"] &= all(u >= s - 0.02 for u, s in sanity.values())
    res[Path(p).name] = r
if control:
    C = json.loads(Path(control).read_text())
    v = [x["pgd_installed"] for k, x in C.items() if k.endswith("|tabayyun_w")]
    res["control_installed_at_loose_eps"] = v
    ok["control"] = max(v) >= 0.20
res["verdict"] = ok
print(json.dumps(res, indent=1))
Path(args[0]).with_name("scorecard_amend15.json").write_text(json.dumps(res, indent=1))
