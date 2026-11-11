"""E07 - component ablations of Tabayyun (Tier A, anchored2 base), descriptive.

Every variant changes ONE component of the base (`tabayyun_anchored2`: k = 2f+1,
gamma = 0.3, nu = 0.05, tau = 50, EMA alpha_P = 0.5, quarantine kept after
promotion, momentum-free promoted tier):

  A4  momentum in M^P (eta_P in {0.5, 0.9}); Thm 2' assumes eta_P = 0, so
      violations of the (momentum-free) certificate are EXPECTED here
  A5  nu, gamma (incl. no cone test), k, promotion period tau, alpha_P
  A10 after-promotion policy for the quarantine: reset / decay

Same worlds and attacks as E02 (dense n=10 and locality n=20, f/n = 0.1).
Seeds 500.. (fresh; ablations are descriptive, not confirmatory).

Usage: python experiments/e07_ablations.py --seeds 10 --seed_offset 500
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun import GateConfig, Gates, TabayyunConfig, TabayyunMemory  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("e02", ROOT / "experiments" / "e02_synthetic_defence.py")
e02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e02)

# name -> (ablation id, overrides of TabayyunConfig, overrides of GateConfig)
VARIANTS = {
    "tab_base":        ("base", {}, {}),
    "tab_etaP0.5":     ("A4", {"eta_P": 0.5}, {}),
    "tab_etaP0.9":     ("A4", {"eta_P": 0.9}, {}),
    "tab_nu0.02":      ("A5", {}, {"nu": 0.02}),
    "tab_nu0.1":       ("A5", {}, {"nu": 0.1}),
    "tab_nu0.2":       ("A5", {}, {"nu": 0.2}),
    "tab_gamma0.1":    ("A5", {}, {"gamma": 0.1}),
    "tab_gamma0.6":    ("A5", {}, {"gamma": 0.6}),
    "tab_nocone":      ("A5", {}, {"cone": False}),
    "tab_k+2":         ("A5", {}, {"k_plus": 2}),
    "tab_tau25":       ("A5", {"tau": 25}, {}),
    "tab_tau100":      ("A5", {"tau": 100}, {}),
    "tab_alphaP0.2":   ("A5", {"alpha_P": 0.2}, {}),
    "tab_alphaP0.9":   ("A5", {"alpha_P": 0.9}, {}),
    "tab_reset":       ("A10", {"after_promotion": "reset"}, {}),
    "tab_decay":       ("A10", {"after_promotion": "decay"}, {}),
}
ATTACKS = ["none", "targeted", "rayn", "edge_of_range", "unbounded"]


def build_variant(defence, wc, f, k, device, alpha_q=None):
    _, cfg_o, gate_o = VARIANTS[defence]
    gate_o = dict(gate_o)
    k = k + gate_o.pop("k_plus", 0)
    gate = GateConfig(**{"k": k, "f": f, "gamma": 0.3, "nu": 0.05, **gate_o})
    aq = e02.ALPHA_Q if alpha_q is None else alpha_q
    cfg = {"n_sources": wc.n_sources, "dk": wc.dk, "dv": wc.dv, "tau": wc.tau,
           "q_gates": Gates(theta=0.5, eta=0.0, alpha=aq), "gate": gate, "basis": "anchored",
           "anchor_min_writers": "f+1", "device": device, **cfg_o}
    return TabayyunMemory(TabayyunConfig(**cfg))


def job(j):
    e02.build = build_variant          # set inside the worker (spawn-safe on Windows)
    r = e02.run_one(j)
    r["ablation"] = VARIANTS[r["defence"]][0]
    r["regime"] = "dense" if r["support"] is None else "locality"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--rhos", default="0.4,0.7")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    variants, rhos = list(VARIANTS), [float(x) for x in a.rhos.split(",")]
    seeds = range(a.seed_offset, a.seed_offset + a.seeds)
    regimes = [(10, None), (20, (-1, -1))]
    if a.smoke:
        variants, rhos, seeds, a.rounds = ["tab_base", "tab_etaP0.9", "tab_reset"], [0.4], range(1), 4
    jobs = [(d, at, 0.1, rho, s, n, a.rounds, "cpu", sup, None)
            for d, at, (n, sup), rho, s in itertools.product(variants, ATTACKS, regimes, rhos, seeds)]
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 400 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "results" / "e07"; out.mkdir(parents=True, exist_ok=True)
    name = "smoke" if a.smoke else f"e07{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    for d, rg, rho in itertools.product(variants, ("dense", "locality"), rhos):
        rr = [r for r in rows if (r["defence"], r["regime"], r["rho"]) == (d, rg, rho)]
        att = [r for r in rr if r["attack"] != "none"]
        worst = lambda m: float(max(np.mean([r[m] for r in att if r["attack"] == x]) for x in ATTACKS[1:]))  # noqa: E731
        summ[f"{rg}|rho={rho}|{d}"] = {
            "ablation": VARIANTS[d][0],
            "clean_recall_l2": float(np.mean([r["recall_l2"] for r in rr if r["attack"] == "none"])),
            "worst_installed": worst("installed"), "worst_flip_l2": worst("asr_l2"),
            "worst_amnesia": worst("amnesia"),
            "violations": int(sum(not r["cert_sound"] for r in rr)),
            "max_drift_over_cert": float(max(r["drift_over_cert_max"] for r in rr)),
            "mean_cert_frob": float(np.mean([r["cert_frob"] for r in rr]))}
    for k, v in summ.items():
        print(f"{k:34} " + " ".join(f"{m}={x:.3f}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = {d: sum(v["violations"] for k, v in summ.items() if k.endswith("|" + d)) for d in variants}
    print("SUMMARY", json.dumps({"rows": len(rows), "violations_by_variant": viol,
                                 "seconds": round(time.time() - t0, 1)}))
    log_run("e07_ablations", vars(a), {"summary": summ, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
