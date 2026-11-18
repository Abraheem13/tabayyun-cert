"""Score E15 (mid-layer memory, certified snapping) against Amendment 20 (M1-M4).

Usage: python scripts/analyze_e15m.py results/e15m/e15m_gpt2xl_confirm_summary.json [more summaries ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

out = {}
for arg in sys.argv[1:]:
    p = Path(arg)
    S = json.loads(p.read_text())
    cells = sorted({"|".join(k.split("|")[:4]) for k in S})
    g = lambda c, d: S.get(f"{c}|{d}")  # noqa: E731
    tab = {}
    for d in ("no_memory", "oracle_value", "titans", "agg_trmean", "agg_median", "agg_geomed", "robustrag",
              "tabayyun_strict", "tabayyun_w", "tabayyun_w_raw"):
        e = [g(c, d) for c in cells if g(c, d)]
        if not e:
            continue
        row = {"final_recall": round(float(np.nanmean([x["final_recall"] for x in e])), 3),
               "worst_final_installed": round(float(np.nanmax([x["worst_final_installed"] for x in e])), 3)}
        if "final_certified_correct_clean" in e[0]:
            row["final_certified_correct"] = round(float(np.mean([x["final_certified_correct_clean"] for x in e])), 3)
            row["worst_final_served_installed"] = round(float(np.nanmax([x["worst_final_served_installed"] for x in e])), 3)
        tab[d] = row
    fv = sum(v.get("final_violations", 0) for v in S.values())
    rv = sum(v.get("read_violations", 0) for v in S.values())
    m2 = [(c, d, g(c, d)["worst_final_installed"]) for c in cells for d in ("tabayyun_strict", "tabayyun_w")
          if g(c, d)["worst_final_installed"] > 0.05]
    m3 = [(c, d, g(c, d)["worst_final_served_installed"]) for c in cells for d in ("tabayyun_strict", "tabayyun_w")
          if g(c, d)["worst_final_served_installed"] > 0.0]
    tf = sum(g(c, "tabayyun_w").get("raw_transfer_failures", 0) for c in cells)
    nc = sum(g(c, "tabayyun_w").get("n_certified", 0) for c in cells)
    well = [c for c in cells if c.endswith("rho=0.7|R=3")]
    m4 = {d: round(float(np.mean([g(c, d)["final_certified_correct_clean"] for c in well])), 3)
          for d in ("tabayyun_strict", "tabayyun_w")}
    oracle_well = round(float(np.mean([g(c, "oracle_value")["final_recall"] for c in well])), 3)
    out[p.stem] = {"table": tab, "M1_final_violations": fv, "M1_read_violations": rv, "M2_exceptions": m2,
                   "M3_served_installed_exceptions": m3,
                   "M4_certified_correct_well_observed": m4, "oracle_edit_success_well_observed": oracle_well,
                   "raw_transfer_failures": [tf, nc],
                   "verdict": {"M1": fv == 0 and rv == 0, "M2": not m2, "M3": not m3,
                               "M4": all(v >= 0.8 * oracle_well for v in m4.values())}}
print(json.dumps(out, indent=1))
Path(sys.argv[1]).with_name("e15m_scorecard.json").write_text(json.dumps(out, indent=1))
