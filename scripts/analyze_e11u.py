"""Score the union adaptive attacker on Tier D (E11 --union) against Amendment 12."""
import json
import sys
from pathlib import Path

for p in sys.argv[1:]:
    S = json.loads(Path(p).read_text())
    cells = sorted({k.rsplit("|", 1)[0] for k in S})
    g = lambda d, m: [S[f"{c}|{d}"][m] for c in cells]  # noqa: E731
    viol = sum(g("tabayyun", "violations")) + sum(g("tabayyun_strict", "violations"))
    b = sum(x <= 0.10 for x in g("tabayyun_strict", "pgd_installed"))
    sanity = {d: {"union_mean": round(sum(g(d, "pgd_installed")) / len(cells), 4),
                  "static_mean": round(sum(g(d, "static_installed")) / len(cells), 4)}
              for d in ("tabayyun", "tabayyun_strict", "agg_trmean")}
    c_ok = all(v["union_mean"] >= v["static_mean"] - 0.02 for v in sanity.values())
    out = {"file": Path(p).name, "cells": len(cells), "H12a_violations": viol,
           "H12b_strict": f"{b}/{len(cells)}", "H12c_attack_sanity": sanity,
           "primary_cells_le_0.10": f"{sum(x <= 0.10 for x in g('tabayyun', 'pgd_installed'))}/{len(cells)}",
           "worst_installed": {d: round(max(g(d, "pgd_installed")), 3) for d in ("tabayyun", "tabayyun_strict", "agg_trmean")},
           "verdict": {"H12a": viol == 0, "H12b": b == len(cells), "H12c": c_ok}}
    print(json.dumps(out, indent=1))
    Path(p).with_name(Path(p).stem.replace("_summary", "") + "_scorecard.json").write_text(json.dumps(out, indent=1))
