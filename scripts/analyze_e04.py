"""Score Tier-B confirmatory runs (E04) against Amendment 3/4.

Usage: python scripts/analyze_e04.py results/e04/e04_<tag>.json [--out NAME]

Per cell (regime, f, rho, R):
  P1  worst installed(method) <= 0.5 * worst installed(Titans)            (bootstrap upper bound)
  C1  Pareto non-domination over (worst installed, worst amnesia, -clean recall)
  C2  [locality] clean recall(method) >= best SECURE baseline (worst installed <= 0.1) + 0.20  (lower bound)
  C3  worst installed(method) <= best USEFUL baseline (recall within 0.05 of method) - 0.50     (upper bound)
  H3  0 certificate / certified-token violations
  H7  [strict variant] |clean recall - Thm 3(b) prediction| <= 0.10
Baselines: titans, agg_*, robustrag, smsr, tma_corr (ablations and the other Tabayyun variant excluded).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "robustrag", "smsr", "tma_corr"]
RNG = np.random.default_rng(0)
B = 4000
TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = json.loads(Path(a.rows).read_text())
    summ = json.loads(Path(a.rows.replace(".json", "_summary.json")).read_text())
    cells = sorted({(r["regime"], r["f"], r["rho"], r.get("R", 1)) for r in rows})
    attacks = sorted({r["attack"] for r in rows if r["attack"] != "none"})
    trials = sorted({r["trial"] for r in rows})
    tix = {t: i for i, t in enumerate(trials)}
    T = len(trials)
    res, tally = {}, defaultdict(lambda: [0, 0])
    for cell in cells:
        rg, f, rho, R = cell
        inst = defaultdict(lambda: np.full((len(attacks), T), np.nan))
        amn = defaultdict(lambda: np.full((len(attacks), T), np.nan))
        rec = defaultdict(lambda: np.full(T, np.nan))
        for r in rows:
            if (r["regime"], r["f"], r["rho"], r.get("R", 1)) != cell:
                continue
            d, t = r["defence"], tix[r["trial"]]
            if r["attack"] == "none":
                rec[d][t] = r["recall"]
            else:
                ai = attacks.index(r["attack"])
                inst[d][ai, t], amn[d][ai, t] = r["installed"], r["amnesia"]
        W = lambda m, i: float(np.nanmax(np.nanmean(m[:, i], 1)))  # noqa: E731  worst attack
        Rc = lambda d, i: float(np.nanmean(rec[d][i]))  # noqa: E731
        all_i = np.arange(T)
        boots = RNG.integers(0, T, size=(B, T))

        def ci(fn):
            v = np.array([fn(b) for b in boots])
            return [fn(all_i), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

        c = {}
        for M in ("tabayyun", "tabayyun_strict"):
            if M not in rec:
                continue
            e = {}
            e["P1"] = ci(lambda i: W(inst[M], i) - 0.5 * W(inst["titans"], i))
            e["P1_pass"] = e["P1"][2] <= TOL
            vec = lambda d: (W(inst[d], all_i), W(amn[d], all_i), -Rc(d, all_i))  # noqa: E731
            m = vec(M)
            dom = [d for d in BASE if all(x <= y + 1e-9 for x, y in zip(vec(d), m)) and vec(d) != m]
            e["C1_dominated_by"], e["C1_pass"] = dom, not dom
            secure = [d for d in BASE if W(inst[d], all_i) <= 0.1]
            bs = max(secure, key=lambda d: Rc(d, all_i)) if secure else None
            e["best_secure"] = bs
            if rg == "locality":
                e["C2"] = ci(lambda i: Rc(M, i) - (Rc(bs, i) if bs else 0.0))
                e["C2_pass"] = e["C2"][1] >= 0.20 - TOL
            useful = [d for d in BASE if Rc(d, all_i) >= Rc(M, all_i) - 0.05]
            bu = min(useful, key=lambda d: W(inst[d], all_i)) if useful else None
            e["best_useful"] = bu
            if bu:
                e["C3"] = ci(lambda i: W(inst[M], i) - W(inst[bu], i))
                e["C3_pass"] = e["C3"][2] <= -0.50 + TOL
            s = summ[f"{rg}|f={f}|rho={rho}|R={R}|{M}"]
            e["H3_violations"] = s["violations"] + s["action_violations"]
            e["H7"] = [Rc(M, all_i), s["H7_pred"]]
            e["H7_pass"] = abs(e["H7"][0] - e["H7"][1]) <= 0.10 + TOL
            e["certified_correct_clean"] = s["certified_correct_clean"]
            e["table"] = {d: {"rec": Rc(d, all_i), "inst": W(inst[d], all_i), "amn": W(amn[d], all_i)}
                          for d in BASE + [M]}
            for k in ("P1_pass", "C1_pass", "C2_pass", "C3_pass", "H7_pass"):
                if k in e:
                    tally[(M, k)][0] += int(bool(e[k])); tally[(M, k)][1] += 1
            tally[(M, "H3_violations")][0] += e["H3_violations"]
            c[M] = e
            fmt = lambda t: f"{t[0]:+.2f}[{t[1]:+.2f},{t[2]:+.2f}]"  # noqa: E731
            print(f"{rg:8} f={f} rho={rho} R={R} {M:15} rec={Rc(M, all_i):.2f} inst={m[0]:.2f} amn={m[1]:.2f} "
                  f"cert={e['certified_correct_clean']:.2f} | P1 {fmt(e['P1'])} {'P' if e['P1_pass'] else 'F'} | "
                  f"C1 {'P' if e['C1_pass'] else 'F:' + ','.join(dom)} | "
                  + (f"C2 {fmt(e['C2'])} vs {bs} {'P' if e['C2_pass'] else 'F'} | " if 'C2' in e else "")
                  + (f"C3 {fmt(e['C3'])} vs {bu} {'P' if e['C3_pass'] else 'F'} | " if 'C3' in e else "")
                  + f"H7 {e['H7'][0]:.2f} vs {e['H7'][1]:.2f} {'P' if e['H7_pass'] else 'F'}")
        res[f"{rg}|f={f}|rho={rho}|R={R}"] = c
    tot = {f"{m}:{k}": (f"{v[0]}/{v[1]}" if k != "H3_violations" else v[0]) for (m, k), v in tally.items()}
    print("SUMMARY", json.dumps(tot))
    out = a.out or a.rows.replace(".json", "_scorecard.json")
    Path(out).write_text(json.dumps({"cells": res, "summary": tot}, indent=1, default=float))


if __name__ == "__main__":
    main()
