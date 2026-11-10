"""Score E02 confirmatory runs against the frozen pre-registration (prereg-v1).

Usage: python scripts/analyze_e02.py [--method tabayyun_anchored2] [--out NAME] results/e02/<files>.json
Writes results/e02/prereg_scorecard.json and prints the scorecard.

Definitions (docs/preregistration.md):
  ASR      = flip in cosine OR in L2 (max of the two per run), worst case over attacks.
  amnesia  = ||read - v_true|| >= 0.5 ||v_true||, worst case over attacks.
  recall   = clean Recall_L2 (attack = none).
  "best robust baseline" is taken separately for every criterion (the strictest choice).
  CIs: paired bootstrap over seeds (10,000 resamples), 95%.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabayyun.certify import promotion_prob  # noqa: E402

ROBUST = ["agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "agg_cclip"]
TAB = "tabayyun_anchored"
TOL = 1e-9   # Amendment 1: thresholds compared with a floating-point tolerance
ATTACKS = ["targeted", "single_shot", "rayn", "edge_of_range", "unbounded"]
RNG = np.random.default_rng(0)
B = 10000


def index(rows):
    """data[(defence, f, rho)][metric] -> array (attack, seed) or (seed,)."""
    seeds = sorted({r["seed"] for r in rows})
    sid = {s: i for i, s in enumerate(seeds)}
    acc = defaultdict(lambda: defaultdict(lambda: np.full((len(ATTACKS), len(seeds)), np.nan)))
    clean = defaultdict(lambda: np.full(len(seeds), np.nan))
    for r in rows:
        key = (r["defence"], r["f_over_n"], r["rho"])
        if r["attack"] == "none":
            clean[key][sid[r["seed"]]] = r["recall_l2"]
        else:
            a = ATTACKS.index(r["attack"])
            acc[key]["asr"][a, sid[r["seed"]]] = max(r["asr"], r["asr_l2"])
            acc[key]["amnesia"][a, sid[r["seed"]]] = r["amnesia"]
            if r.get("installed") is not None:
                acc[key]["installed"][a, sid[r["seed"]]] = r["installed"]
    return acc, clean, len(seeds)


def worst(mat, idx):
    """Worst attack (max over attacks of the mean over the resampled seeds)."""
    return np.nanmax(np.nanmean(mat[:, idx], axis=1))


def boot(fn, n):
    idx = RNG.integers(0, n, size=(B, n))
    vals = np.array([fn(i) for i in idx])
    point = fn(np.arange(n))
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(path, regime):
    rows = json.loads(Path(path).read_text())
    acc, clean, n = index(rows)
    defs = sorted({r["defence"] for r in rows})
    cells = sorted({(r["f_over_n"], r["rho"]) for r in rows})
    out = {"regime": regime, "cells": {}, "summary": {}}
    for f, rho in cells:
        c = {}
        tab = (TAB, f, rho)
        get_asr = lambda d, i: worst(acc[(d, f, rho)]["asr"], i)  # noqa: E731
        get_amn = lambda d, i: worst(acc[(d, f, rho)]["amnesia"], i)  # noqa: E731
        get_rec = lambda d, i: float(np.nanmean(clean[(d, f, rho)][i]))  # noqa: E731
        c["table"] = {d: {"asr_worst": get_asr(d, np.arange(n)), "amnesia_worst": get_amn(d, np.arange(n)),
                          "clean_recall_l2": get_rec(d, np.arange(n))} for d in defs}
        # P1: Tabayyun worst ASR <= 0.5 * Titans worst ASR
        c["P1"] = boot(lambda i: get_asr(TAB, i) - 0.5 * get_asr("titans", i), n)
        c["P1_pass"] = c["P1"][2] <= 0 + TOL
        # best baseline per criterion
        c["best_asr"] = min(ROBUST, key=lambda d: get_asr(d, np.arange(n)))
        c["best_amnesia"] = min(ROBUST, key=lambda d: get_amn(d, np.arange(n)))
        c["best_recall"] = max(ROBUST, key=lambda d: get_rec(d, np.arange(n)))
        d_asr = boot(lambda i: get_asr(TAB, i) - get_asr(c["best_asr"], i), n)
        d_amn = boot(lambda i: get_amn(TAB, i) - get_amn(c["best_amnesia"], i), n)
        d_rec = boot(lambda i: get_rec(TAB, i) - get_rec(c["best_recall"], i), n)
        c["diff_vs_best"] = {"asr": d_asr, "amnesia": d_amn, "recall_l2": d_rec}
        if regime == "dense":
            c["P2_pass"] = d_asr[2] <= 0.05 + TOL and d_rec[1] >= -0.02 - TOL
        else:
            c["P2_pass"] = d_rec[1] >= 0.20 - TOL and d_amn[2] <= -0.20 + TOL and d_asr[2] <= 0.0 + TOL
            nc = ("tab_nocorr", f, rho)
            if nc in clean:
                a2 = boot(lambda i: max(get_asr("tab_nocorr", i), get_amn("tab_nocorr", i))
                          - max(get_asr(TAB, i), get_amn(TAB, i)), n)
                c["A2"] = a2
                c["A2_pass"] = a2[1] >= 0.20 - TOL
            if ("tabayyun", f, rho) in clean:
                a6 = boot(lambda i: get_rec("tabayyun", i) - get_rec(TAB, i), n)
                c["A6"] = a6
                c["A6_pass"] = a6[2] <= -0.20 + TOL
        # secondary (Amendment 1): attacker value actually installed, worst attack
        inst = acc[tab].get("installed")
        if inst is not None and not np.all(np.isnan(inst)):
            c["installed_worst"] = {d: float(worst(acc[(d, f, rho)]["installed"], np.arange(n)))
                                    for d in defs if "installed" in acc[(d, f, rho)]}
        # H7 (Amendment 1): Thm 3(b) predicts clean Recall_L2 of the method
        nn = rows[0]["n"]
        ff = max(1, round(f * nn))
        k = 2 * ff + 1
        ms = [nn - ff] if regime == "dense" else list(range(k, (nn - ff) // 2 + 1))
        R = max(r_["rounds"] for r_ in rows) if "rounds" in rows[0] else 10
        c["H7_pred"] = float(np.mean([promotion_prob(m, rho, R - 1, k) for m in ms]))
        c["H7_obs"] = get_rec(TAB, np.arange(n))
        c["H7_pass"] = abs(c["H7_obs"] - c["H7_pred"]) <= 0.10 + TOL
        out["cells"][f"f={f}|rho={rho}"] = c
    cs = out["cells"].values()
    out["summary"] = {k: f"{sum(bool(c[k]) for c in cs if k in c)}/{sum(k in c for c in cs)}"
                      for k in ["P1_pass", "P2_pass", "A2_pass", "A6_pass", "H7_pass"]}
    tabrows = [r for r in rows if "cert_sound" in r]
    out["summary"]["H3_violations"] = f"{sum(not r['cert_sound'] for r in tabrows)}/{len(tabrows)}"
    return out


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"


def main():
    global TAB
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["results/e02/e02_dense.json", "results/e02/e02_locality.json"])
    ap.add_argument("--method", default="tabayyun_anchored")
    ap.add_argument("--out", default="prereg_scorecard")
    a = ap.parse_args()
    TAB, paths = a.method, a.paths
    res = {}
    for p in paths:
        regime = "locality" if "locality" in p else "dense"
        res[regime] = s = score(p, regime)
        print(f"\n===== {regime.upper()}  ({p})")
        for cell, c in s["cells"].items():
            t = c["table"]
            print(f"\n{cell}")
            for d, v in t.items():
                print(f"  {d:18} ASR_worst={v['asr_worst']:.2f}  amnesia_worst={v['amnesia_worst']:.2f}"
                      f"  clean_RecallL2={v['clean_recall_l2']:.2f}")
            dv = c["diff_vs_best"]
            print(f"  P1 (tab - 0.5*titans ASR) {fmt(c['P1'])} -> {'PASS' if c['P1_pass'] else 'FAIL'}")
            print(f"  vs best: ASR {fmt(dv['asr'])} ({c['best_asr']}), amnesia {fmt(dv['amnesia'])} "
                  f"({c['best_amnesia']}), recall {fmt(dv['recall_l2'])} ({c['best_recall']})"
                  f" -> P2 {'PASS' if c['P2_pass'] else 'FAIL'}")
            print(f"  H7 Thm3b: observed recall {c['H7_obs']:.2f} vs predicted {c['H7_pred']:.2f} -> "
                  f"{'PASS' if c['H7_pass'] else 'FAIL'}")
            if "installed_worst" in c:
                print("  installed (secondary):", {d: round(v, 2) for d, v in c["installed_worst"].items()})
            if "A2" in c:
                print(f"  A2 nocorr-tab {fmt(c['A2'])} -> {'PASS' if c['A2_pass'] else 'FAIL'};"
                      f"  A6 identity-tab recall {fmt(c['A6'])} -> {'PASS' if c['A6_pass'] else 'FAIL'}")
        print("\nSUMMARY", regime, json.dumps(s["summary"]))
    Path(f"results/e02/{a.out}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
