"""E09 - limits and failure modes of Tabayyun (Tier A).  Pre-declared negative regimes.

Part H4  (negative regime, pre-registered in v1): a fraction of facts is known to
         ONE honest source only (unique_frac), or observation is very heterogeneous
         (Dirichlet 0.1).  Such facts can never gather k witnesses, so the certified
         (high-risk) read must lose them.  We also measure what the owner still gets
         from its own quarantine (low-risk read), and that security is unaffected.
Part SYB (out of threat model): the gate assumes f, the world has f_act >= f adversaries.
Part UPS (out of threat model, independence violated): f real adversaries plus c honest
         sources whose UPSTREAM FEED is poisoned for the attacked fact (they relay v*
         in good faith).  The certificate's reference counts them as honest.
Part LIV (liveness): MedianShift, a denial-of-promotion attack on the cone test.

Usage: python experiments/e09_limits.py --seeds 10 --seed_offset 700
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun.attacks import MedianShift  # noqa: E402
from tabayyun.data.streams import FactWorld, WorldConfig, run_round  # noqa: E402
from tabayyun.metrics import amnesia, installed, l2_flip, l2_recall  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("e02", ROOT / "experiments" / "e02_synthetic_defence.py")
e02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e02)

T = "tabayyun_anchored2"
BASE = ["titans", "agg_trmean", "agg_median", "robustrag", T]


def _world(n, n_adv, rho, seed, **kw):
    return WorldConfig(n_sources=n, n_adv=n_adv, dk=32, dv=32, n_facts=24, rho=rho, sigma=0.1, tau=50,
                       seed=seed, **kw)


def one(job):
    part, defence, attack, rho, seed, p = job
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    rounds = 10
    res = {"part": part, "defence": defence, "attack": attack, "rho": rho, "seed": seed, **p}
    if part == "H4":
        n, f = 10, 1
        wc = _world(n, f, rho, seed, unique_frac=p["unique_frac"], dirichlet=p["dirichlet"])
    elif part == "SYB":
        n, f = 10, 2
        wc = _world(n, p["f_act"], rho, seed)
    elif part == "UPS":
        n, f = 10, 2
        wc = _world(n, f, rho, seed)
    else:                                                    # LIV
        n = p["n"]
        f = max(1, int(round(p["frac"] * n)))
        kw = {} if p["regime"] == "dense" else {"support_min": 2 * f + 1, "support_max": (n - f) // 2}
        wc = _world(n, f, rho, seed, **kw)
    k = 2 * f + 1
    w = FactWorld(wc)
    shared_idx = torch.nonzero(~w.unique).flatten()
    tgt = int(shared_idx[0])                                 # the attacked fact is always a shared one
    k_star, v_true = w.keys[tgt], w.values[tgt]
    v_star = -v_true
    honest = w.honest
    if part == "UPS" and p["c"] > 0:                         # poisoned upstream feed for c honest sources
        relays = w.honest[: p["c"]]
        w.bias[relays, tgt] = v_star - v_true                # they now observe v* for the attacked fact
    mem = e02.build(defence, wc, f, k, "cpu")
    watk, hook = (None, None)
    if attack == "median_shift":
        if hasattr(mem, "delta_hook"):
            hook = MedianShift(w.adv, seed=seed)
    else:
        watk, hook = e02.make_attack(attack, w, mem, k_star, v_star)
    if hook is not None:
        mem.delta_hook = hook
    tab = defence.startswith("tab")
    if tab:
        mem.oracle_honest = honest
    sound, worst_ratio = True, 0.0
    for _ in range(rounds):
        run_round(mem, w, watk)
        if tab:
            td, ce = mem.true_drift(), mem.certificate
            sound &= td <= ce * (1 + 1e-4) + 1e-5
            if td > 1e-6:
                worst_ratio = max(worst_ratio, td / max(ce, 1e-12))
    others = torch.tensor([j for j in range(wc.n_facts) if j != tgt])
    sh = others[~w.unique[others]]
    un = others[w.unique[others]]
    res.update({"recall_all": l2_recall(mem, w.keys[others], w.values[others]),
                "recall_shared": l2_recall(mem, w.keys[sh], w.values[sh]),
                "recall_unique": l2_recall(mem, w.keys[un], w.values[un]) if len(un) else None,
                "amnesia": float(amnesia(mem, k_star, v_true)),
                "installed": float(installed(mem, k_star, v_star)) if attack not in ("none", "median_shift") else None,
                "flip": float(l2_flip(mem, k_star, v_star, v_true)) if attack not in ("none", "median_shift") else None})
    if tab:
        res.update({"cert_sound": bool(sound), "drift_over_cert": worst_ratio,
                    "promoted_frac": float(mem.last.promoted.float().mean()) if mem.last is not None else 0.0})
        if len(un):                                          # what the owner still reads at low risk
            res["recall_unique_owner_lowrisk"] = l2_recall(mem, w.keys[un], w.values[un], src=w.owner[un],
                                                           high_risk=False)
    return res


def jobs_for(seeds, smoke):
    J = []
    rhos = [0.7] if smoke else [0.4, 0.7]
    for rho, s in itertools.product(rhos, seeds):
        for uf, dr in [(0.0, float("inf")), (0.25, float("inf")), (0.5, float("inf")), (0.0, 0.1)]:
            for d, at in itertools.product(BASE, ["none", "edge_of_range"]):
                J.append(("H4", d, at, rho, s, {"unique_frac": uf, "dirichlet": dr}))
        for fa in [1, 2, 3, 4, 5]:
            for d, at in itertools.product(BASE, ["targeted", "edge_of_range", "unbounded"]):
                J.append(("SYB", d, at, rho, s, {"f_act": fa}))
        for c in [0, 1, 2, 3]:
            for d in BASE:
                J.append(("UPS", d, "edge_of_range", rho, s, {"c": c}))
        for n, rg, fr in [(10, "dense", 0.1), (10, "dense", 0.2), (10, "dense", 0.3),
                          (20, "locality", 0.1), (20, "locality", 0.15)]:
            for at in ["none", "median_shift"]:
                J.append(("LIV", T, at, rho, s, {"n": n, "regime": rg, "frac": fr}))
    return J


def mean(rr, m):
    v = [r[m] for r in rr if r.get(m) is not None and not (isinstance(r[m], float) and np.isnan(r[m]))]
    return round(float(np.mean(v)), 3) if v else None


def summarise(rows):
    S = {"H4": {}, "SYB": {}, "UPS": {}, "LIV": {}}
    sel = lambda part, **kw: [r for r in rows if r["part"] == part and all(r[a] == b for a, b in kw.items())]  # noqa: E731
    for uf, dr in [(0.0, float("inf")), (0.25, float("inf")), (0.5, float("inf")), (0.0, 0.1)]:
        name = f"unique={uf}" if dr == float("inf") else "dirichlet=0.1"
        for d in BASE:
            cl = sel("H4", defence=d, attack="none", unique_frac=uf, dirichlet=dr)
            at = sel("H4", defence=d, attack="edge_of_range", unique_frac=uf, dirichlet=dr)
            e = {"recall_all": mean(cl, "recall_all"), "recall_shared": mean(cl, "recall_shared"),
                 "recall_unique": mean(cl, "recall_unique"), "installed": mean(at, "installed"),
                 "flip": mean(at, "flip")}
            if d == T:
                e["recall_unique_owner_lowrisk"] = mean(cl, "recall_unique_owner_lowrisk")
                e["violations"] = int(sum(not r["cert_sound"] for r in cl + at))
            S["H4"][f"{name}|{d}"] = e
    for fa in [1, 2, 3, 4, 5]:
        for d in BASE:
            rr = sel("SYB", defence=d, f_act=fa)
            worst = lambda m: max(mean([r for r in rr if r["attack"] == a], m) for a in  # noqa: E731
                                  ["targeted", "edge_of_range", "unbounded"])
            e = {"worst_installed": worst("installed"), "worst_flip": worst("flip"), "worst_amnesia": worst("amnesia")}
            if d == T:
                e["violation_rate"] = round(float(np.mean([not r["cert_sound"] for r in rr])), 3)
                e["max_drift_over_cert"] = round(float(max(r["drift_over_cert"] for r in rr)), 2)
            S["SYB"][f"f_act={fa}|{d}"] = e
    for c in [0, 1, 2, 3]:
        for d in BASE:
            rr = sel("UPS", defence=d, c=c)
            e = {"installed": mean(rr, "installed"), "flip": mean(rr, "flip"), "amnesia": mean(rr, "amnesia")}
            if d == T:
                e["violation_rate"] = round(float(np.mean([not r["cert_sound"] for r in rr])), 3)
            S["UPS"][f"c={c}|{d}"] = e
    for n, rg, fr in [(10, "dense", 0.1), (10, "dense", 0.2), (10, "dense", 0.3), (20, "locality", 0.1),
                      (20, "locality", 0.15)]:
        for rho in sorted({r["rho"] for r in rows}):
            cl = sel("LIV", attack="none", n=n, regime=rg, frac=fr, rho=rho)
            at = sel("LIV", attack="median_shift", n=n, regime=rg, frac=fr, rho=rho)
            S["LIV"][f"{rg}|f/n={fr}|rho={rho}"] = {
                "recall_clean": mean(cl, "recall_all"), "recall_median_shift": mean(at, "recall_all"),
                "amnesia_median_shift": mean(at, "amnesia"), "promoted_clean": mean(cl, "promoted_frac"),
                "promoted_median_shift": mean(at, "promoted_frac"),
                "violations": int(sum(not r["cert_sound"] for r in cl + at))}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=700)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = range(1) if a.smoke else range(a.seed_offset, a.seed_offset + a.seeds)
    jobs = jobs_for(seeds, a.smoke)
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    for r in rows:                                           # JSON-safe
        if r.get("dirichlet") == float("inf"):
            r["dirichlet"] = "inf"
    out = ROOT / "results" / "e09"; out.mkdir(parents=True, exist_ok=True)
    name = "smoke" if a.smoke else f"e09{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    for r in rows:
        if r.get("dirichlet") == "inf":
            r["dirichlet"] = float("inf")
    S = summarise(rows)
    for part, tab in S.items():
        print(f"--- {part}")
        for kk, v in tab.items():
            print(f"{kk:40} " + " ".join(f"{m}={x}" for m, x in v.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(S, indent=1))
    print("SUMMARY", json.dumps({"rows": len(rows), "seconds": round(time.time() - t0, 1)}))
    log_run("e09_limits", vars(a), {"summary": S},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
