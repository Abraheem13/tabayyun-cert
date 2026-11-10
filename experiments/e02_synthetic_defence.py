"""E02 - Hypotheses 2, 3, 6 on the controlled multi-source testbed.

Grid: defence x attack x f/n x rho (observation rate) x seed.
The same omniscient adversary is used against every quarantine-based
defence: promotion-level attacks rewrite the adversarial rows of the
quarantine states for Tabayyun *and* for the robust-aggregation baselines.

Usage:
    python experiments/e02_synthetic_defence.py --smoke
    python experiments/e02_synthetic_defence.py --seeds 10 --workers 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabayyun import GateConfig, Gates, TabayyunConfig, TabayyunMemory  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Rayn, TargetedWrite, Unbounded  # noqa: E402
from tabayyun.baselines import (HuberSharedMemory, IsolateAggregateMemory, RobustAggMemory,  # noqa: E402
                                SharedMemory, SMSRMemory)
from tabayyun.data import FactWorld, WorldConfig, run_round  # noqa: E402
from tabayyun.metrics import amnesia, benign_recall, installed, l2_flip, l2_recall, target_cos, targeted_success  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DEFENCES = ["titans", "titans_clip", "titans_ttl", "agg_mean", "agg_trmean", "agg_median",
            "agg_krum", "agg_geomed", "agg_cclip", "tab_nocorr", "tabayyun", "tabayyun_anchored",
            "tabayyun_anchored2", "tabayyun_anchored2_add", "agg_median_anc", "agg_trmean_anc", "agg_krum_anc",
            "miras_yaad", "smsr", "robustrag", "tma_corr"]
ATTACKS = ["none", "targeted", "single_shot", "rayn", "edge_of_range", "unbounded"]
QUARANTINE = ("agg_", "tab")


ALPHA_Q = 0.005    # forget gate of quarantines and shared memory; tuned on CLEAN runs only (Day 3)


def build(defence, wc: WorldConfig, f, k, device, alpha_q=None):
    aq = ALPHA_Q if alpha_q is None else alpha_q
    gates = Gates(theta=0.5, eta=0.0, alpha=aq)
    n, dk, dv, tau = wc.n_sources, wc.dk, wc.dv, wc.tau
    if defence == "titans":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=aq), device=device)
    if defence == "titans_clip":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=aq, clip=0.5), device=device)
    if defence == "titans_ttl":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.0, alpha=0.2), device=device)
    if defence == "miras_yaad":
        return HuberSharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=aq), delta=0.1, device=device)
    if defence == "smsr":
        return SMSRMemory(n, dk, dv, gates, seed=wc.seed, device=device)
    if defence == "robustrag":
        return IsolateAggregateMemory(n, dk, dv, gates, device=device)
    if defence == "tma_corr":
        # TMA-NM-style corroboration-gated elevation: >= k distinct writers, plain mean,
        # no cone test, no trimming (hence no certificate); same anchored basis as ours
        gate = GateConfig(k=k, f=f, nu=0.05, cone=False, trim=False)
        return TabayyunMemory(TabayyunConfig(n_sources=n, dk=dk, dv=dv, tau=tau, q_gates=gates, gate=gate,
                                             basis="anchored", anchor_min_writers="f+1", device=device))
    if defence.startswith("agg_"):
        anc = defence.endswith("_anc")
        return RobustAggMemory(n, dk, dv, gates, agg=defence[4:].removesuffix("_anc"), f=f, tau=tau,
                               device=device, basis="anchored" if anc else "identity")
    if defence.startswith("tab"):
        gate = GateConfig(k=k, f=f, gamma=0.3, nu=0.05, corroborate=(defence != "tab_nocorr"))
        return TabayyunMemory(TabayyunConfig(n_sources=n, dk=dk, dv=dv, tau=tau, q_gates=gates, gate=gate,
                                             basis="anchored" if defence.startswith("tabayyun_anchored") else "identity",
                                             anchor_min_writers="f+1" if defence.startswith("tabayyun_anchored2") else "k",
                                             rule="additive" if defence.endswith("_add") else "ema",
                                             alpha_P=0.1 if defence.endswith("_add") else 0.5,
                                             device=device))
    raise ValueError(defence)


def make_attack(name, w: FactWorld, mem, k_star, v_star):
    """Returns (write_attack, promotion_hook)."""
    if name == "none":
        return None, None
    if name == "targeted":
        return TargetedWrite(w.adv, k_star, v_star), None
    if name == "single_shot":
        return TargetedWrite(w.adv, k_star, v_star, scale=10.0, steps=[0]), None
    if name == "rayn":
        return Rayn(w.adv, k_star, v_star, eps=0.1), None
    quarantined = isinstance(mem, (TabayyunMemory, RobustAggMemory, SMSRMemory, IsolateAggregateMemory))
    if name == "edge_of_range":
        if quarantined:
            return TargetedWrite(w.adv, k_star, v_star), EdgeOfRange(w.adv, w.honest, k_star, v_star, nu=0.05)
        return TargetedWrite(w.adv, k_star, v_star), None
    if name == "unbounded":
        if quarantined:
            return TargetedWrite(w.adv, k_star, v_star), Unbounded(w.adv, k_star, v_star)
        return TargetedWrite(w.adv, k_star, v_star, scale=10.0), None
    raise ValueError(name)


def run_one(job):
    defence, attack, fr, rho, seed, n, rounds, device, support, alpha_q = job
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    f = max(1, int(round(fr * n)))
    k = 2 * f + 1
    smin, smax = (None, None) if support is None else support
    if smin is not None and smin == -1:            # "k..n_h/2": minority-held facts
        smin, smax = k, (n - f) // 2
        if smin > smax:
            raise ValueError(f"minority regime needs k <= (n-f)//2; n={n}, f={f}, k={k}")
    wc = WorldConfig(n_sources=n, n_adv=f, dk=32, dv=32, n_facts=24, rho=rho, sigma=0.1, tau=50, seed=seed,
                     support_min=smin, support_max=smax)
    w = FactWorld(wc, device=device)
    mem = build(defence, wc, f, k, device, alpha_q)
    k_star, v_true = w.keys[0], w.values[0]
    v_star = -v_true                                                   # flip target
    watk, hook = make_attack(attack, w, mem, k_star, v_star)
    if hook is not None:
        mem.delta_hook = hook
    certified = defence.startswith("tab")
    if certified:
        mem.oracle_honest = w.honest
    sound, worst_ratio = True, 0.0
    for _ in range(rounds):
        run_round(mem, w, watk)
        if certified:
            td, ce = mem.true_drift(), mem.certificate
            sound &= td <= ce * (1 + 1e-4) + 1e-5
            if td > 1e-6:
                worst_ratio = max(worst_ratio, td / max(ce, 1e-12))
    benign = torch.arange(1, wc.n_facts)
    shared = benign[~w.unique[1:]]
    res = {"defence": defence, "attack": attack, "f_over_n": fr, "rho": rho, "seed": seed, "n": n,
           "support": None if support is None else [smin, smax],
           "asr": float(targeted_success(mem, k_star, v_star, v_true)) if attack != "none" else None,
           "asr_l2": float(l2_flip(mem, k_star, v_star, v_true)) if attack != "none" else None,
           "amnesia": float(amnesia(mem, k_star, v_true)),
           "installed": float(installed(mem, k_star, v_star)) if attack != "none" else None,
           "target_cos": target_cos(mem, k_star, v_star),
           "recall": benign_recall(mem, w.keys[shared], w.values[shared], extra_candidates=v_star[None]),
           "recall_l2": l2_recall(mem, w.keys[shared], w.values[shared])}
    if certified:
        U = mem.basis()
        res.update({"cert_sound": bool(sound), "cert_frob": mem.certificate, "true_drift": mem.true_drift(),
                    "drift_over_cert_max": worst_ratio, "n_directions": int(U.shape[0]),
                    "qcert_kstar": float(mem.query_certificate(k_star[None])[0]) if U.shape[0] else 0.0,
                    "qdrift_kstar": float(((mem.MP - mem.MP_ref) @ k_star).norm()),
                    "promoted_frac": float(mem.last.promoted.float().mean()) if mem.last is not None else 0.0})
    return res


def summarise(rows, defs, atks, fracs, rhos):
    table = {}
    for d, fr, rho in itertools.product(defs, fracs, rhos):
        rr = [r for r in rows if r["defence"] == d and r["f_over_n"] == fr and r["rho"] == rho]
        att = [r for r in rr if r["attack"] != "none"]
        cl = [r for r in rr if r["attack"] == "none"]
        key = f"{d}|f={fr}|rho={rho}"
        worst = lambda m: float(max(np.mean([r[m] for r in att if r["attack"] == a])  # noqa: E731
                                    for a in atks if a != "none"))
        table[key] = {"asr_mean": float(np.mean([r["asr"] for r in att])) if att else None,
                      "asr_worst": worst("asr") if att else None,
                      "asr_l2_worst": worst("asr_l2") if att else None,
                      "amnesia_worst": worst("amnesia") if att else None,
                      "clean_recall": float(np.mean([r["recall"] for r in cl])) if cl else None,
                      "clean_recall_l2": float(np.mean([r["recall_l2"] for r in cl])) if cl else None,
                      "attacked_recall_l2": float(np.mean([r["recall_l2"] for r in att])) if att else None}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu is faster for these tiny per-step ops")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--defences", default=",".join(DEFENCES))
    ap.add_argument("--attacks", default=",".join(ATTACKS))
    ap.add_argument("--rhos", default="0.2,0.4,0.7")
    ap.add_argument("--fracs", default="0.1,0.2,0.3")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--support", default="all", choices=["all", "minority"],
                    help="minority: each fact is known to m ~ U[k, (n-f)//2] honest sources only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--alpha_q", type=float, default=ALPHA_Q)
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="confirmatory runs use seeds 100.. (exploratory probes used 0..9)")
    ap.add_argument("--out", default="results/e02")
    a = ap.parse_args()
    if a.smoke:
        defs, atks = ["titans", "agg_trmean", "tabayyun", "tabayyun_anchored"], ["none", "targeted", "edge_of_range"]
        fracs, rhos, seeds, rounds = [0.2], [0.4], range(1), 3
    else:
        defs, atks = a.defences.split(","), a.attacks.split(",")
        fracs = [float(x) for x in a.fracs.split(",")]
        rhos = [float(x) for x in a.rhos.split(",")]
        seeds, rounds = range(a.seed_offset, a.seed_offset + a.seeds), a.rounds
    n = a.n
    support = None if a.support == "all" else (-1, -1)
    jobs = [(d, at, fr, rho, s, n, rounds, a.device, support, a.alpha_q)
            for d, at, fr, rho, s in itertools.product(defs, atks, fracs, rhos, seeds)]
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(run_one, jobs, chunksize=4)):
                rows.append(r)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    else:
        rows = [run_one(j) for j in jobs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / ("smoke.json" if a.smoke else f"e02{('_' + a.tag) if a.tag else ''}.json")).write_text(json.dumps(rows, indent=1))
    table = summarise(rows, defs, atks, fracs, rhos)
    for kk, v in table.items():
        print(kk, json.dumps(v))
    tab = [r for r in rows if "cert_sound" in r]
    metrics = {"n_runs": len(rows), "seconds": round(time.time() - t0, 1),
               "cert_violations": sum(not r["cert_sound"] for r in tab),
               "cert_runs": len(tab),
               "max_drift_over_cert": max([r["drift_over_cert_max"] for r in tab], default=None),
               "table": table}
    print("SUMMARY", json.dumps({k: v for k, v in metrics.items() if k != "table"}))
    log_run("e02_synthetic_defence", {k: v for k, v in vars(a).items()}, metrics,
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
