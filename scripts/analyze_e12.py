"""Score E12 (witness complexity) against Amendment 14 (W1-W5)."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/e12/e12_amend14_summary.json")
S = json.loads(p.read_text())
cfgs = sorted({tuple(int(x.split("=")[1]) for x in k.split("|")[:2]) for k in S})
g = lambda n, f, d: S[f"n={n}|f={f}|{d}"]  # noqa: E731
MAJ = ["agg_mean", "agg_trmean", "agg_median", "agg_geomed", "robustrag"]
TAB = ["tabayyun_anchored2", "tabayyun_w"]
out = {"thresholds": {f"n={n},f={f}": {d: g(n, f, d)["witness_threshold"] for d in
                                       ["titans", "tma_corr"] + MAJ + TAB} for n, f in cfgs}}
out["W1_cone_threshold_is_2f+1"] = f"{sum(g(n, f, 'tabayyun_anchored2')['witness_threshold'] == 2 * f + 1 for n, f in cfgs)}/{len(cfgs)}"
out["W2_agree_threshold_is_f+1"] = f"{sum(g(n, f, 'tabayyun_w')['witness_threshold'] == f + 1 for n, f in cfgs)}/{len(cfgs)}"
big = [(n, f) for n, f in cfgs if n >= 20]
w3 = [(n, f, d) for n, f in big for d in MAJ
      if g(n, f, d)["witness_threshold"] is not None and g(n, f, d)["witness_threshold"] < 0.4 * n]
out["W3_majority_threshold_ge_0.4n"] = f"{len(big) * len(MAJ) - len(w3)}/{len(big) * len(MAJ)}"
out["W3_exceptions"] = w3
w4 = {d: {"worst_installed": max(g(n, f, d)["worst_installed"] for n, f in cfgs),
          "worst_flip": max(g(n, f, d)["worst_flip"] for n, f in cfgs),
          "worst_amnesia_when_served": max(g(n, f, d)["worst_amnesia_served"] for n, f in cfgs),
          "violations": sum(g(n, f, d)["violations"] for n, f in cfgs),
          "max_drift_over_cert": max(g(n, f, d)["max_drift_over_cert"] for n, f in cfgs)} for d in TAB}
out["W4_security"] = w4
out["W5_forgeable"] = {d: f"{sum(g(n, f, d)['worst_flip'] >= 0.5 for n, f in cfgs)}/{len(cfgs)}" for d in ("titans", "tma_corr")}
out["baseline_worst_flip"] = {d: max(g(n, f, d)["worst_flip"] for n, f in cfgs) for d in MAJ}
out["baseline_worst_amnesia_when_served"] = {d: max(g(n, f, d)["worst_amnesia_served"] for n, f in cfgs) for d in MAJ}
N = len(cfgs)
out["verdict"] = {"W1": out["W1_cone_threshold_is_2f+1"] == f"{N}/{N}", "W2": out["W2_agree_threshold_is_f+1"] == f"{N}/{N}",
                  "W3": not w3,
                  "W4": all(v["worst_installed"] <= 0.05 and v["worst_flip"] <= 0.05 and v["worst_amnesia_when_served"] <= 0.10
                            and v["violations"] == 0 for v in w4.values()),
                  "W5": all(v == f"{N}/{N}" for v in out["W5_forgeable"].values())}
print(json.dumps(out, indent=1))
p.with_name("scorecard_amend14.json").write_text(json.dumps(out, indent=1))
