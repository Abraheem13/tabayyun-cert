"""Score E13 (capacity and cost scaling) against Amendment 17 (S1, S2, S5) and print the scaling table."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
S = json.loads(p.read_text())
keys = [k.split("|") for k in S]
Fs = sorted({int(k[0][2:]) for k in keys})
cells = sorted({"|".join(k[1:5]) for k in keys})
g = lambda F, c, d: S[f"F={F}|{c}|{d}"]  # noqa: E731
out = {"table": {}, "verdict": {}}
viol = sum(v.get("violations", 0) for v in S.values())
s2 = [(F, c, d, g(F, c, d)["worst_installed"]) for F in Fs for c in cells for d in ("tabayyun_strict", "tabayyun_w")
      if g(F, c, d)["worst_installed"] > 0.05]
s5 = {}
for F in Fs:
    rw = sum(g(F, c, "tabayyun_w")["clean_recall"] for c in cells) / len(cells)
    rs = sum(g(F, c, "tabayyun_strict")["clean_recall"] for c in cells) / len(cells)
    s5[F] = (round(rw, 3), round(rs, 3))
    row = {}
    for d in ("titans", "agg_trmean", "agg_median", "agg_geomed", "robustrag", "tabayyun_strict", "tabayyun_w"):
        e = [g(F, c, d) for c in cells]
        row[d] = {"recall": round(sum(x["clean_recall"] for x in e) / len(e), 3),
                  "worst_installed": round(max(x["worst_installed"] for x in e), 3),
                  "seconds_combine": round(sum(x["seconds_combine"] for x in e) / len(e), 4)}
        if "certified_correct_clean" in e[0]:
            row[d]["certified_correct"] = round(sum(x["certified_correct_clean"] for x in e) / len(e), 3)
    row["seconds_write_quarantines"] = round(sum(g(F, c, "tabayyun_w")["seconds_write"] for c in cells) / len(cells), 3)
    out["table"][F] = row
out["S1_violations"] = viol
out["S2_exceptions"] = s2
out["S5_recall_W_vs_strict"] = s5
out["verdict"] = {"S1": viol == 0, "S2": not s2, "S5": all(w >= s - 0.05 for w, s in s5.values())}
# Amendment 18 (sharded): certified-and-correct in the two well-observed cells (rho = 0.7, R = 3)
well = [c for c in cells if c.endswith("rho=0.7|R=3")]
sh3 = {F: {d: round(sum(g(F, c, d)["certified_correct_clean"] for c in well) / len(well), 3)
           for d in ("tabayyun_strict", "tabayyun_w")} for F in Fs}
out["SH3_certified_correct_well_observed"] = sh3
out["verdict"]["SH3"] = all(v >= 0.80 for row in sh3.values() for v in row.values())
print(json.dumps(out, indent=1))
p.with_name(p.stem.replace("_summary", "") + "_scorecard.json").write_text(json.dumps(out, indent=1))
