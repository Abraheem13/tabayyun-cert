"""Score E05 (adaptive PGD/BPDA) against Amendment 5 criteria H8a-c."""
import json
import sys
from pathlib import Path

for p in sys.argv[1:]:
    S = json.loads(Path(p).read_text())
    cells = sorted({k.rsplit("|", 1)[0] for k in S})
    viol = sum(S[f"{c}|{m}"]["violations"] for c in cells for m in ("tabayyun", "tabayyun_strict"))
    b = sum(S[f"{c}|tabayyun_strict"]["pgd_installed"] <= 0.10 for c in cells)
    b_prim = sum(S[f"{c}|tabayyun"]["pgd_installed"] <= 0.10 for c in cells)
    fails_c = []
    for c in cells:
        ref = min(S[f"{c}|agg_{a}"]["pgd_installed"] for a in ("trmean", "median", "geomed"))
        if S[f"{c}|tabayyun_strict"]["pgd_installed"] > ref + 0.05:
            fails_c.append(c)
    worst = lambda m: max(S[f"{c}|{m}"]["pgd_installed"] for c in cells)  # noqa: E731
    out = {"file": Path(p).name, "cells": len(cells), "H8a_violations": viol,
           "H8b_strict": f"{b}/{len(cells)}", "H8b_primary_descr": f"{b_prim}/{len(cells)}",
           "H8c": f"{len(cells) - len(fails_c)}/{len(cells)}", "H8c_fails": fails_c,
           "worst_pgd": {m: round(worst(m), 3) for m in
                         ("tabayyun", "tabayyun_strict", "agg_trmean", "agg_median", "agg_geomed", "agg_krum", "robustrag")}}
    out["verdict"] = "PASS" if viol == 0 and b == len(cells) and not fails_c else "FAIL"
    print(json.dumps(out, indent=1))
    Path(p).with_name(Path(p).stem.replace("_summary", "") + "_scorecard.json").write_text(json.dumps(out, indent=1))
