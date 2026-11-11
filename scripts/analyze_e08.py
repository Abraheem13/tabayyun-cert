"""Score E08 (Tier C, O-RAN) against Amendment 8 criteria H10a-e."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/e08/e08_confirm_summary.json")
S = json.loads(p.read_text())
T = "tabayyun_anchored2"
cells = sorted({k.rsplit("|", 1)[0] for k in S})
base = sorted({k.rsplit("|", 1)[1] for k in S} - {T})
obj = lambda e: (e["worst_hijack"], e["worst_regret"], e["clean_regret"])  # noqa: E731
res = {"H10a_violations": 0, "H10b": 0, "H10c": 0, "H10e": 0}
cert, doms = [], {}
for c in cells:
    t = S[f"{c}|{T}"]
    res["H10a_violations"] += t["violations"] + t["action_violations"]
    res["H10b"] += t["worst_hijack"] <= 0.05
    d = [b for b in base if all(x <= y for x, y in zip(obj(S[f"{c}|{b}"]), obj(t))) and obj(S[f"{c}|{b}"]) != obj(t)]
    res["H10c"] += not d
    if d:
        doms[c] = d
    res["H10e"] += S[f"{c}|titans"]["worst_hijack"] >= 0.5
    cert.append(t["certified_clean"])
n = len(cells)
out = {"H10a_violations": res["H10a_violations"], "H10b": f"{res['H10b']}/{n}", "H10c": f"{res['H10c']}/{n}",
       "H10c_dominated_by": doms, "H10d_certified_10pct": round(sum(cert) / n, 3),
       "H10e": f"{res['H10e']}/{n}"}
out["verdict"] = {"H10a": res["H10a_violations"] == 0, "H10b": res["H10b"] == n, "H10c": res["H10c"] == n,
                  "H10d": sum(cert) / n >= 0.60, "H10e": res["H10e"] == n}
print(json.dumps(out, indent=1))
p.with_name("scorecard_amend8.json").write_text(json.dumps(out, indent=1))
