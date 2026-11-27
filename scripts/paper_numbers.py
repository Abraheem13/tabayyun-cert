"""Regenerate every number, table row and plot data file of the TIFS manuscript from results/.

Usage: python scripts/paper_numbers.py   ->  paper/tifs/data/*.dat and paper/tifs/data/numbers.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tifs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.loads((R / p).read_text())  # noqa: E731
N = {}

# ---------------------------------------------------------------- witness complexity (E12, Amendment 14)
W = J("e12/scorecard_amend14.json")["thresholds"]
S12 = J("e12/e12_amend14_summary.json")
rows = []
for n in (10, 20, 40, 80):
    t = W[f"n={n},f=2"]
    rows.append([n, t["agg_trmean"], t["agg_median"], t["agg_geomed"], t["robustrag"], t["tabayyun_anchored2"], t["tabayyun_w"]])
(OUT / "witness_f2.dat").write_text("n trmean median geomed robustrag tabc tabw\n" +
                                    "\n".join(" ".join(map(str, r)) for r in rows) + "\n")
N["witness"] = W
N["witness_viol"] = sum(v.get("violations", 0) for v in S12.values())
N["witness_max_drift_over_cert"] = max(v.get("max_drift_over_cert", 0) for v in S12.values())

# ---------------------------------------------------------------- static protocol, Tiers B and D (Amendment 16)
DEFS = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "robustrag", "smsr", "tma_corr",
        "tabayyun", "tabayyun_strict", "tabayyun_w"]
static = {}
for tag, f in [("d128", "e04/e04_d128_w_confirm_summary.json"), ("d256", "e04/e04_d256_w_confirm_summary.json"),
               ("d512", "e04/e04_d512_w_confirm_summary.json"), ("gpt2xl", "e10/e10_gpt2xl_w_confirm_summary.json"),
               ("qwen15b", "e10/e10_qwen15b_w_confirm_summary.json")]:
    S = J(f)
    cells = sorted({k.rsplit("|", 1)[0] for k in S})
    dn = [c for c in cells if c.startswith("dense")]
    lc = [c for c in cells if c.startswith("locality")]
    static[tag] = {}
    for d in DEFS:
        e = {"rec_dense": float(np.mean([S[f"{c}|{d}"]["clean_recall"] for c in dn])),
             "rec_loc": float(np.mean([S[f"{c}|{d}"]["clean_recall"] for c in lc])),
             "inst_worst": float(max(S[f"{c}|{d}"]["worst_installed"] for c in cells))}
        if "certified_correct_clean" in S[f"{cells[0]}|{d}"]:
            e["cert_correct"] = float(np.mean([S[f"{c}|{d}"]["certified_correct_clean"] for c in cells]))
            e["viol"] = int(sum(S[f"{c}|{d}"]["violations"] + S[f"{c}|{d}"]["action_violations"] for c in cells))
            e["served_inst_worst"] = float(max(S[f"{c}|{d}"].get("worst_served_installed", 0.0) for c in cells))
        static[tag][d] = e
N["static"] = static

# ---------------------------------------------------------------- worst case over every attack run
def worst(files, key, defs):
    out = {}
    for f in files:
        if not (R / f).exists():
            continue
        S = J(f)
        for k, v in S.items():
            d = k.rsplit("|", 1)[1]
            if d in defs and key in v:
                out[d] = max(out.get(d, 0.0), float(v[key]))
    return out

ADV = ["tabayyun", "tabayyun_strict", "tabayyun_w", "agg_trmean", "agg_median", "agg_geomed", "agg_krum", "robustrag"]
# static column: every confirmatory static run (Amendments 3/10 trials 200-299 and 800-899, and 1100-1199)
_eB = worst([f"e04/e04_{t}_confirm_summary.json" for t in ("d128", "d256", "d512")], "worst_installed", DEFS)
_eD = worst([f"e10/e10_{t}_confirm_summary.json" for t in ("gpt2xl", "qwen15b")], "worst_installed", DEFS)
tierB_static = {d: max([static[t][d]["inst_worst"] for t in ("d128", "d256", "d512")] + [_eB.get(d, 0.0)]) for d in DEFS}
tierD_static = {d: max([static[t][d]["inst_worst"] for t in ("gpt2xl", "qwen15b")] + [_eD.get(d, 0.0)]) for d in DEFS}
tierB_pgd = worst(["e05/e05_d128_confirm_summary.json", "e05/e05_d256_confirm_summary.json"], "pgd_installed", ADV)
tierB_union = worst([f"e05/e05_{t}_union_summary.json" for t in ("d128", "d256", "d512")] +
                    [f"e05/e05_{t}_w_summary.json" for t in ("d128", "d256", "d512")], "pgd_installed", ADV)
tierD_pgd = worst(["e11/e11_gpt2xl_confirm_summary.json", "e11/e11_qwen15b_confirm_summary.json"], "pgd_installed", ADV)
tierD_union = worst([f"e11/e11_{t}_union_summary.json" for t in ("gpt2xl", "qwen15b")] +
                    [f"e11/e11_{t}_w_union_summary.json" for t in ("gpt2xl", "qwen15b")], "pgd_installed", ADV)
N["worst"] = {"B": {"static": tierB_static, "pgd": tierB_pgd, "union": tierB_union},
              "D": {"static": tierD_static, "pgd": tierD_pgd, "union": tierD_union}}
# served-installed under certify-or-abstain, union attacker
N["served_union"] = {"B": worst([f"e05/e05_{t}_w_summary.json" for t in ("d128", "d256", "d512")], "served_installed", ADV),
                     "D": worst([f"e11/e11_{t}_w_union_summary.json" for t in ("gpt2xl", "qwen15b")], "served_installed", ADV)}
amn = {}
for t in ("d128", "d256", "d512"):
    S = J(f"e05/e05_{t}_w_summary.json")
    for d in ("tabayyun_strict", "tabayyun_w"):
        amn.setdefault(d, []).append(float(np.mean([v["amnesia_mean"] for k, v in S.items() if k.endswith("|" + d)])))
        N.setdefault("served_correct_B", {}).setdefault(d, []).append(
            float(np.mean([v["served_correct"] for k, v in S.items() if k.endswith("|" + d)])))
N["union_amnesia_B"] = amn
N["control_eps2"] = J("e05/e05_d128_wcontrol_summary.json")

# per-model union worst for the bar chart (Tier B, Day 20 run)
bar = []
for t in ("d128", "d256", "d512"):
    S = J(f"e05/e05_{t}_w_summary.json")
    w = {d: max(v["pgd_installed"] for k, v in S.items() if k.endswith("|" + d))
         for d in ("tabayyun", "tabayyun_strict", "tabayyun_w", "agg_trmean")}
    bar.append([t, w["agg_trmean"], w["tabayyun"], w["tabayyun_strict"], w["tabayyun_w"]])
(OUT / "union_tierB.dat").write_text("model trmean tabc0 tabc tabw\n" + "\n".join(" ".join(map(str, r)) for r in bar) + "\n")

# ---------------------------------------------------------------- scaling (Amendments 17 and 18)
def scaling(f):
    S = J(f)
    Fs = sorted({int(k.split("|")[0][2:]) for k in S})
    out = []
    for F in Fs:
        well = [k for k in S if k.startswith(f"F={F}|") and "rho=0.7|R=3" in k]
        allc = [k for k in S if k.startswith(f"F={F}|")]
        g = lambda d, m, ks: [S[k][m] for k in ks if k.endswith("|" + d)]  # noqa: E731
        out.append({"F": F,
                    "cc_C": float(np.mean(g("tabayyun_strict", "certified_correct_clean", well))),
                    "cc_W": float(np.mean(g("tabayyun_w", "certified_correct_clean", well))),
                    "inst_C": float(max(g("tabayyun_strict", "worst_installed", allc))),
                    "inst_W": float(max(g("tabayyun_w", "worst_installed", allc))),
                    "inst_TrMean": float(max(g("agg_trmean", "worst_installed", allc))),
                    "rec_C": float(np.mean(g("tabayyun_strict", "clean_recall", allc))),
                    "rec_W": float(np.mean(g("tabayyun_w", "clean_recall", allc))),
                    "rec_TrMean": float(np.mean(g("agg_trmean", "clean_recall", allc))),
                    "t_W": float(np.mean(g("tabayyun_w", "seconds_combine", allc))),
                    "t_TrMean": float(np.mean(g("agg_trmean", "seconds_combine", allc))),
                    "viol": int(sum(S[k].get("violations", 0) for k in allc))})
    return out

N["scale_flat"] = scaling("e13/e13_gpt2xl_scale_summary.json")
N["scale_shard"] = scaling("e13/e13_gpt2xl_sharded_summary.json")
(OUT / "scaling.dat").write_text("F ccCflat ccWflat ccCshard ccWshard instCflat instWflat instCshard instWshard\n" + "\n".join(
    f"{a['F']} {a['cc_C']:.4f} {a['cc_W']:.4f} {b['cc_C']:.4f} {b['cc_W']:.4f} {a['inst_C']:.4f} {a['inst_W']:.4f} "
    f"{b['inst_C']:.4f} {b['inst_W']:.4f}" for a, b in zip(N["scale_flat"], N["scale_shard"])) + "\n")

# ---------------------------------------------------------------- federation (Amendment 6) and O-RAN (Amendment 8)
S = J("e06/e06_confirm_summary.json")
fed = {}
for k, v in S.items():
    rg, fc, rho, d = k.split("|")
    e = fed.setdefault(d, {"rec_loc": [], "flip": 0.0, "inst": 0.0, "amn": 0.0, "viol": 0})
    if rg == "locality":
        e["rec_loc"].append(v["clean_recall"])
    e["flip"] = max(e["flip"], v["worst_flip"]); e["inst"] = max(e["inst"], v["worst_installed"])
    e["amn"] = max(e["amn"], v["worst_amnesia"]); e["viol"] += v.get("violations", 0)
N["fed"] = {d: {**e, "rec_loc": float(np.mean(e["rec_loc"]))} for d, e in fed.items()}
S = J("e08/e08_confirm_summary.json")
oran = {}
for k, v in S.items():
    rg, f_, rho, d = k.split("|")
    e = oran.setdefault(d, {"hijack": [], "regret_clean": [], "regret_worst": [], "loc_hijack": []})
    e["hijack"].append(v["worst_hijack"]); e["regret_clean"].append(v["clean_regret"]); e["regret_worst"].append(v["worst_regret"])
    if rg == "locality":
        e["loc_hijack"].append(v["worst_hijack"])
N["oran"] = {d: {"hijack_max": max(e["hijack"]), "loc_hijack_max": max(e["loc_hijack"]),
                 "regret_clean": float(np.mean(e["regret_clean"])), "regret_worst_max": max(e["regret_worst"])}
             for d, e in oran.items()}
N["oran_card"] = J("e08/scorecard_amend8.json")

# ---------------------------------------------------------------- certificate audit: rows and violations of every confirmatory file
audit = {"rows": 0, "tab_rows": 0, "violations": 0, "out_of_model_rows": 0, "out_of_model_violations": 0}


def out_of_model(r):
    """Runs outside the threat model or the design, by pre-registration: momentum in M^P (A4 ablation,
    Amendment 7) and more adversaries than the gate assumes (Sybils beyond f, Amendment 9)."""
    return str(r.get("defence", "")).startswith("tab_etaP") or (r.get("part") == "SYB" and (r.get("f_act") or 0) > 2)
for f in sorted(R.glob("e0[2-9]/*.json")) + sorted(R.glob("e1[0-4]/*.json")) + sorted(R.glob("e15m/*.json")):
    if any(x in f.name for x in ("summary", "scorecard", "smoke", "tuning", "explore", "prereg")) or f.name.startswith("e01"):
        continue
    if f.parent.name == "e03":
        continue
    try:
        rows_ = json.loads(f.read_text())
    except Exception:
        continue
    if not isinstance(rows_, list):
        continue
    audit["rows"] += len(rows_)
    for r in rows_:
        if not isinstance(r, dict):
            continue
        if "cert_sound" in r or "violations" in r:
            v = int(r.get("cert_sound") is False) + int(r.get("violations", 0) or 0) + int(r.get("action_violations", 0) or 0)
            if out_of_model(r):
                audit["out_of_model_rows"] += 1; audit["out_of_model_violations"] += v
            else:
                audit["tab_rows"] += 1; audit["violations"] += v
        if "final_violations" in r:                      # E15: identity of the LM output (Proposition 1)
            audit["output_rows"] = audit.get("output_rows", 0) + 1
            audit["output_violations"] = audit.get("output_violations", 0) + int(r["final_violations"])
            audit["output_certified"] = audit.get("output_certified", 0) + int(round(r["certified_frac"] * 16))
N["audit"] = audit

(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
print(json.dumps({k: N[k] for k in ("audit", "witness_viol", "worst", "served_union", "union_amnesia_B")}, indent=1, default=float))


# ---------------------------------------------------------------- LaTeX table bodies (generated, never typed by hand)
NAME = {"titans": "Titans (ungated)", "agg_mean": "Mean (FedAvg)", "agg_trmean": "Trimmed mean",
        "agg_median": "Coordinate median", "agg_krum": "Krum", "agg_geomed": "Geometric median",
        "robustrag": "RobustRAG-style", "smsr": "SMSR-style", "tma_corr": "TMA-NM-style",
        "tabayyun": "Tabayyun-C0 (ablation)", "tabayyun_strict": "Tabayyun-C", "tabayyun_w": "Tabayyun-W"}
f2 = lambda x: f"{x:.2f}"  # noqa: E731
f3 = lambda x: f"{x:.3f}"  # noqa: E731
lines = []
for d in DEFS:
    B = [static[t][d] for t in ("d128", "d256", "d512")]
    D = [static[t][d] for t in ("gpt2xl", "qwen15b")]
    cc = lambda L: f2(np.mean([e["cert_correct"] for e in L])) if "cert_correct" in L[0] else "--"  # noqa: E731
    row = [NAME[d], f2(np.mean([e["rec_dense"] for e in B])), f2(np.mean([e["rec_loc"] for e in B])),
           f3(max(e["inst_worst"] for e in B)), cc(B),
           f2(np.mean([e["rec_dense"] for e in D])), f2(np.mean([e["rec_loc"] for e in D])),
           f3(max(e["inst_worst"] for e in D)), cc(D)]
    if d == "tabayyun":
        lines.append("\\midrule")
    lines.append(" & ".join(row) + " \\\\")
(OUT / "tab_static.tex").write_text("\n".join(lines) + "\n")

lines = []
for d in ["agg_trmean", "agg_median", "agg_geomed", "agg_krum", "robustrag", "tabayyun", "tabayyun_strict", "tabayyun_w"]:
    cell = []
    for tier in ("B", "D"):
        w = N["worst"][tier]
        vals = [w["static"].get(d), w["pgd"].get(d), w["union"].get(d)]
        cell += [f3(v) if v is not None else "--" for v in vals]
        cell.append(f3(max(v for v in vals if v is not None)))
    if d == "tabayyun":
        lines.append("\\midrule")
    lines.append(" & ".join([NAME[d]] + cell) + " \\\\")
(OUT / "tab_worst.tex").write_text("\n".join(lines) + "\n")

WD = ["titans", "tma_corr", "agg_trmean", "agg_median", "agg_geomed", "robustrag", "tabayyun_anchored2", "tabayyun_w"]
lines = []
for cfg in ["n=10,f=2", "n=20,f=2", "n=40,f=2", "n=80,f=2", "n=40,f=1", "n=40,f=3"]:
    n, f = [int(x.split("=")[1]) for x in cfg.split(",")]
    t = W[cfg]
    lines.append(" & ".join([f"{n}", f"{f}"] + [str(t[d]) for d in WD] + [str(f + 1), str(2 * f + 1)]) + " \\\\")
(OUT / "tab_witness.tex").write_text("\n".join(lines) + "\n")

FN = {"fedavg_titans": "FedAvg over Titans", "fed_mean": "Mean", "fed_trmean": "Trimmed mean", "fed_median": "Median",
      "fed_krum": "Krum", "fed_geomed": "Geometric median", "fed_tabayyun": "Tabayyun-C (two-level)"}
lines = []
for d in ["fedavg_titans", "fed_mean", "fed_trmean", "fed_median", "fed_krum", "fed_geomed", "fed_tabayyun"]:
    e = N["fed"][d]
    lines.append(" & ".join([FN[d], f2(e["rec_loc"]), f2(e["flip"]), f2(e["amn"])]) + " \\\\")
(OUT / "tab_fed.tex").write_text("\n".join(lines) + "\n")
ON = {"titans": "Titans (ungated)", "miras_yaad": "Miras-Yaad (Huber)", "agg_trmean": "Trimmed mean",
      "agg_median": "Coordinate median", "agg_geomed": "Geometric median", "agg_krum": "Krum",
      "robustrag": "RobustRAG-style", "smsr": "SMSR-style", "tma_corr": "TMA-NM-style", "tabayyun_anchored2": "Tabayyun-C"}
lines = []
for d in ["titans", "miras_yaad", "agg_trmean", "agg_median", "agg_geomed", "agg_krum", "robustrag", "smsr", "tma_corr",
          "tabayyun_anchored2"]:
    e = N["oran"][d]
    lines.append(" & ".join([ON[d], f2(e["hijack_max"]), f3(e["regret_clean"]), f2(e["regret_worst_max"])]) + " \\\\")
(OUT / "tab_oran.tex").write_text("\n".join(lines) + "\n")
print("tables written")


# ---------------------------------------------------------------- recall law: predicted vs measured (Amendment 16 files)
rl = ["pred meas variant"]
for f in ["e04/e04_d128_w_confirm_summary.json", "e04/e04_d256_w_confirm_summary.json", "e04/e04_d512_w_confirm_summary.json",
          "e10/e10_gpt2xl_w_confirm_summary.json", "e10/e10_qwen15b_w_confirm_summary.json"]:
    S = J(f)
    for k, v in S.items():
        d = k.rsplit("|", 1)[1]
        if d in ("tabayyun_strict", "tabayyun_w"):
            rl.append(f"{v['H7_pred']:.4f} {v['clean_recall']:.4f} {'W' if d == 'tabayyun_w' else 'C'}")
(OUT / "recall_law.dat").write_text("\n".join(rl) + "\n")
N["recall_law_maxabs"] = {var: max(abs(float(a) - float(b)) for a, b, c in (l.split() for l in rl[1:]) if c == var)
                          for var in ("C", "W")}

# ---------------------------------------------------------------- Tier D trade-off: locality recall vs worst installed over every attack run
tr = ["x y label adaptive"]
LBL = {"titans": "Titans", "agg_mean": "Mean", "agg_trmean": "TrMean", "agg_median": "Median", "agg_krum": "Krum",
       "agg_geomed": "GeoMed", "robustrag": "RobustRAG", "smsr": "SMSR", "tma_corr": "TMA-NM",
       "tabayyun": "C0", "tabayyun_strict": "C", "tabayyun_w": "W"}
wD = N["worst"]["D"]
for d in DEFS:
    vals = [wD["static"].get(d), wD["pgd"].get(d), wD["union"].get(d)]
    x = max(v for v in vals if v is not None)
    y = float(np.mean([static[t][d]["rec_loc"] for t in ("gpt2xl", "qwen15b")]))
    tr.append(f"{x:.4f} {y:.4f} {LBL[d]} {int(wD['pgd'].get(d) is not None or wD['union'].get(d) is not None)}")
(OUT / "tradeoff_D.dat").write_text("\n".join(tr) + "\n")
print("recall-law max |pred-meas|:", N["recall_law_maxabs"])
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
ANCH = {"Titans": "west", "Mean": "south", "TrMean": "south", "Median": "east", "Krum": "west", "GeoMed": "north",
        "RobustRAG": "west", "SMSR": "east", "TMA-NM": "west", "C0": "west", "C": "west", "W": "west"}
for flag, name in (("1", "tradeoff_D_adapt.dat"), ("0", "tradeoff_D_static.dat")):
    rows_ = [l.split() for l in tr[1:] if l.split()[3] == flag]
    (OUT / name).write_text("x y label anchor\n" + "\n".join(f"{r[0]} {r[1]} {r[2]} {ANCH[r[2]]}" for r in rows_) + "\n")


# ---------------------------------------------------------------- paired bootstrap CIs, locality clean recall (Tabayyun-W minus baseline)
from collections import defaultdict  # noqa: E402
rng = np.random.default_rng(0)
CI = {}
for name, f in [("MemLM d128", "e04/e04_d128_w_confirm.json"), ("MemLM d256", "e04/e04_d256_w_confirm.json"),
                ("MemLM d512", "e04/e04_d512_w_confirm.json"), ("GPT-2 XL", "e10/e10_gpt2xl_w_confirm.json"),
                ("Qwen2.5-1.5B", "e10/e10_qwen15b_w_confirm.json")]:
    rec = defaultdict(dict)
    for r in J(f):
        if r["attack"] == "none" and r["regime"] == "locality" and r["recall"] == r["recall"]:
            rec[(r["f"], r["rho"], r["R"], r["trial"])][r["defence"]] = r["recall"]
    keys = [k for k, v in rec.items() if all(d in v for d in ("tabayyun_w", "tabayyun_strict", "robustrag", "agg_trmean"))]
    CI[name] = {}
    for b in ("tabayyun_strict", "robustrag", "agg_trmean"):
        dlt = np.array([rec[k]["tabayyun_w"] - rec[k][b] for k in keys])
        bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)]
        CI[name][b] = (float(dlt.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(dlt))
c = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
(OUT / "tab_ci.tex").write_text("\n".join(f"{nm} & {c(v['tabayyun_strict'])} & {c(v['robustrag'])} & {c(v['agg_trmean'])} \\\\"
                                          for nm, v in CI.items()) + "\n")
N["ci_locality"] = CI
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))


# ---------------------------------------------------------------- Tier R (real sources, E14) and Tier M (mid-layer memory, E15)
E14 = J("e14/e14_confirm_scorecard.json")["table"]
M = {t: J(f"e15m/e15m_{t}_confirm_summary.json") for t in ("gpt2xl", "qwen15b")}


def tierm(t, d, k):
    v = [e for kk, e in M[t].items() if kk.endswith("|" + d)]
    if k == "final_recall":
        return float(np.nanmean([e[k] for e in v]))
    return float(np.nanmax([e[k] for e in v]))


RM = [("vote", "Plurality vote"), ("titans", "Titans (ungated)"), ("agg_mean", "Mean (FedAvg)"), ("agg_trmean", "Trimmed mean"),
      ("agg_median", "Coordinate median"), ("agg_geomed", "Geometric median"), ("robustrag", "RobustRAG-style"),
      ("tabayyun_strict", "Tabayyun-C"), ("tabayyun_w", "Tabayyun-W")]
lines, NR = [], {}
for d, nm in RM:
    acc = np.mean([E14[f][d]["clean_accuracy"] for f in E14])
    ins = max(E14[f][d]["worst_installed"] for f in E14)
    row = [nm, f2(acc), f3(ins)]
    for t in ("gpt2xl", "qwen15b"):
        row += (["--", "--"] if d == "vote" else [f2(tierm(t, d, "final_recall")), f3(tierm(t, d, "worst_final_installed"))])
    lines.append(" & ".join(row) + r" \\")
    NR[d] = row[1:]
(OUT / "tab_real.tex").write_text("\n".join(lines) + "\n")
N["tier_r_m"] = NR
N["tier_r"] = {f: {d: E14[f][d] for d in ("vote", "agg_trmean", "tabayyun_strict", "tabayyun_w")} for f in E14}
N["tier_m"] = {t: {"oracle": tierm(t, "oracle_value", "final_recall"), "none": tierm(t, "no_memory", "final_recall"),
                   "cc_W": float(np.mean([e["final_certified_correct_clean"] for k, e in M[t].items() if k.endswith("|tabayyun_w")])),
                   "cc_C": float(np.mean([e["final_certified_correct_clean"] for k, e in M[t].items() if k.endswith("|tabayyun_strict")])),
                   "raw_W": tierm(t, "tabayyun_w_raw", "final_recall")} for t in M}
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
print(json.dumps({"audit": N["audit"], "tier_m": N["tier_m"]}, indent=1))
