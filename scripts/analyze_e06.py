"""Score E06 (federation) against Amendment 6 criteria H9a-d."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/e06/e06_confirm_summary.json"
S = json.loads(path.read_text())
cells = sorted({k.rsplit("|", 1)[0] for k in S})
base = ["fed_mean", "fed_trmean", "fed_median", "fed_krum", "fed_geomed", "fedavg_titans"]
obj = lambda e: (e["worst_flip"], e["worst_amnesia"], -e["clean_recall"])  # noqa: E731
res = {"H9a": 0, "H9b": 0, "H9c": 0, "H9d": 0}
for c in cells:
    t = S[f"{c}|fed_tabayyun"]
    res["H9a"] += t["violations"] == 0
    res["H9b"] += t["worst_installed"] <= 0.05 and t["worst_flip"] <= 0.10
    dominated = any(all(b <= a for a, b in zip(obj(t), obj(S[f"{c}|{d}"]))) and obj(S[f"{c}|{d}"]) != obj(t)
                    for d in base)
    res["H9c"] += not dominated
    res["H9d"] += S[f"{c}|fedavg_titans"]["worst_flip"] >= 0.5
out = {k: f"{v}/{len(cells)}" for k, v in res.items()}
out["verdict"] = "PASS" if all(v == len(cells) for v in res.values()) else "FAIL"
print(json.dumps(out))
(path.parent / "scorecard_amend6.json").write_text(json.dumps(out, indent=1))
