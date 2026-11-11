"""E08 - Tier C: O-RAN slicing control from real ColO-RAN KPI traces.

A memory-augmented xApp chooses, for one base station, the RAN slicing action
a = (scheduling policy, slice-PRB configuration) out of 3 x 28 = 84 actions.
Its test-time memory maps an action key k_a to the KPI profile it expects
(per slice: DL throughput, log DL buffer, granted PRBs; z-scored), and it acts by
    a_hat = argmax_a  w . M k_a          (linear SLA utility w).
Sources are KPI reporters (E2 nodes / monitoring xApps).  Every honest report
is a REAL measurement window from the ColO-RAN dataset (Polese et al., IEEE TMC
2022; data/oran/coloran_kpi.npz built by scripts/prep_coloran.py).  Compromised
reporters spoof KPIs so that the worst action (the one that starves URLLC and
eMBB) looks best.

Regimes: dense (every honest reporter has measured every action) and locality
(each action measured by m ~ U[k, (n-f)//2] honest reporters).
Metrics per control epoch (one promotion round):
  hijack   chosen action == attacker's target
  regret   u(a*) - u(a_hat) in true utility (z units), a* = true best action
  opt_frac (u(a_hat) - u_min) / (u* - u_min)
  certified (Tabayyun): Thm 3(a') certified epsilon-optimality from per-coordinate
            per-query bounds: eps_hat >= regret of a_hat under the honest
            reference M~^P; certified = eps_hat <= 10% of the utility range
            (certified_exact = eps_hat = 0, the strict Thm 3(a) form).
            Soundness (action_violations) = reference regret > eps_hat.

Usage: python experiments/e08_oran.py --trials 20 --trial_offset 600 --tag confirm
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
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun import Gates  # noqa: E402
from tabayyun.certify import certified_regret  # noqa: E402
from tabayyun.data.streams import run_round  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("e02", ROOT / "experiments" / "e02_synthetic_defence.py")
e02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e02)

DEFS = ["titans", "titans_clip", "miras_yaad", "agg_mean", "agg_trmean", "agg_median", "agg_geomed",
        "agg_krum", "agg_median_anc", "robustrag", "smsr", "tma_corr", "tabayyun_anchored2"]
ATTACKS = ["none", "targeted", "rayn", "edge_of_range", "unbounded"]
# SLA utility on z-scored features [eMBB thr, eMBB buf, eMBB prb, MTC thr, MTC buf, MTC prb, URLLC thr, URLLC buf, URLLC prb]
W_UTIL = torch.tensor([1.0, 0, 0, 0, -0.5, 0, 0, -1.0, 0])
_BANK = None
THETA = 0.05    # write step of every memory (all defences); set from CLEAN runs only (scripts/tune_e08_clean.py)


def bank():
    global _BANK
    if _BANK is None:
        X = np.load(ROOT / "data" / "oran" / "coloran_kpi.npz")["X"]          # (7,3,28,5,8,9)
        mu = np.nanmean(np.nanmean(X, axis=(3, 4)), axis=(0, 1, 2))
        sd = np.nanstd(np.nanmean(X, axis=(3, 4)), axis=(0, 1, 2)) + 1e-9
        _BANK = torch.tensor((X - mu) / sd / 3.0, dtype=torch.float32)          # z / 3: ||v|| ~ 1
    return _BANK


class ORANWorld:
    """Same interface as FactWorld (keys, honest, adv, round_schedule, honest_value)."""

    def __init__(self, bs, n, f, rho, tau, regime, seed, dk=96):
        g = torch.Generator().manual_seed(seed)
        B = bank()[bs].reshape(84, 5 * 8, 9)                                   # action x sample x feat
        self.valid = ~torch.isnan(B).any(-1)                                    # (84, 40)
        self.samples = torch.nan_to_num(B)
        self.values = torch.stack([self.samples[a][self.valid[a]].mean(0) for a in range(84)])
        self.cfg = SimpleNamespace(n_sources=n, n_adv=f, dk=dk, dv=9, tau=tau, seed=seed)
        self.dev, self.g, self.rho, self.F = "cpu", g, rho, 84
        Q, _ = torch.linalg.qr(torch.randn(dk, dk, generator=g))
        self.keys = Q[:, :84].T.contiguous()
        self.adv, self.honest = torch.arange(f), torch.arange(f, n)
        k = 2 * f + 1
        self.relevant = torch.zeros(n, 84, dtype=torch.bool)
        for a in range(84):
            if regime == "dense":
                srcs = self.honest
            else:
                m = int(torch.randint(k, (n - f) // 2 + 1, (1,), generator=g))
                srcs = self.honest[torch.randperm(len(self.honest), generator=g)[:m]]
            self.relevant[srcs, a] = True
        self.u = self.values @ W_UTIL
        self.best, self.worst = int(self.u.argmax()), int(self.u.argmin())

    def round_schedule(self):
        sched = [[] for _ in range(self.cfg.n_sources)]
        for s in self.honest.tolist():
            obs = (torch.rand(84, generator=self.g) < self.rho) & self.relevant[s]
            fs = torch.nonzero(obs).flatten()
            fs = fs[torch.randperm(len(fs), generator=self.g)].tolist()
            sched[s] = fs[: self.cfg.tau]
        return sched

    def honest_value(self, s, a):
        idx = torch.nonzero(self.valid[a]).flatten()
        j = idx[int(torch.randint(len(idx), (1,), generator=self.g))]
        return self.samples[a, j]


def decide(mem, w: ORANWorld, reader):
    Y = reader(w.keys)                                                          # (84, 9)
    util = Y @ W_UTIL
    return int(util.argmax()), Y, util


def _gates(**kw):
    return Gates(**{**kw, "theta": THETA})


def one(job):
    regime, n, f, rho, attack, defence, seed, rounds = job[:8]
    global THETA
    if len(job) > 8:
        THETA = job[8]
    e02.Gates = _gates                     # every defence gets the same write step THETA
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    bs = seed % 7
    w = ORANWorld(bs, n, f, rho, 50, regime, seed)
    k = 2 * f + 1
    mem = e02.build(defence, w.cfg, f, k, "cpu")
    target = w.worst
    k_star = w.keys[target]
    v_star = w.values[w.best] + 0.5 * W_UTIL / W_UTIL.norm()                    # spoofed "excellent" KPIs
    watk, hook = e02.make_attack(attack, w, mem, k_star, v_star)
    if hook is not None:
        mem.delta_hook = hook
    tab = defence.startswith("tab")
    if tab:
        mem.oracle_honest = w.honest
    u_rng = float(w.u[w.best] - w.u.min())
    ep = []
    sound, act_viol = True, 0
    for r in range(rounds):
        run_round(mem, w, watk)
        a_hat, Y, util = decide(mem, w, lambda q: mem.read(q))
        e = {"hijack": float(a_hat == target), "regret": float(w.u[w.best] - w.u[a_hat]),
             "opt_frac": float((w.u[a_hat] - w.u.min()) / u_rng)}
        if tab:
            td, ce = mem.true_drift(), mem.certificate
            sound &= td <= ce * (1 + 1e-4) + 1e-5
            U = mem.basis()
            Bc = mem.drift.query_coord_bound(w.keys, U) if U.shape[0] else torch.zeros(84, 9)
            eps_hat = certified_regret(util, Bc, W_UTIL)
            u_ref = (w.keys @ mem.MP_ref.T) @ W_UTIL
            if float(u_ref.max() - u_ref[a_hat]) > eps_hat * (1 + 1e-4) + 1e-5:
                act_viol += 1
            e["certified"] = float(eps_hat <= 0.1 * u_rng)        # certified 10%-optimal
            e["certified_exact"] = float(eps_hat == 0.0)
            e["eps_rel"] = eps_hat / u_rng
        ep.append(e)
    fin = ep[-1]
    res = {"regime": regime, "n": n, "f": f, "rho": rho, "attack": attack, "defence": defence, "seed": seed,
           "bs": bs, "hijack": fin["hijack"], "regret": fin["regret"], "opt_frac": fin["opt_frac"],
           "hijack_any": float(max(e["hijack"] for e in ep)),
           "regret_mean": float(np.mean([e["regret"] for e in ep]))}
    if tab:
        res.update({"cert_sound": bool(sound), "action_violations": act_viol, "certified": fin["certified"],
                    "certified_exact": fin["certified_exact"], "eps_rel": fin["eps_rel"],
                    "certified_mean": float(np.mean([e["certified"] for e in ep])),
                    "cert_frob": mem.certificate, "true_drift": mem.true_drift()})
    return res


CELLS = [("dense", 10, 1), ("dense", 10, 2), ("locality", 20, 2), ("locality", 20, 3)]


def summarise(rows, rhos):
    summ = {}
    for (rg, n, f), rho, d in itertools.product(CELLS, rhos, DEFS):
        rr = [r for r in rows if (r["regime"], r["f"], r["rho"], r["defence"]) == (rg, f, rho, d)]
        if not rr:
            continue
        att = [r for r in rr if r["attack"] != "none"]
        cl = [r for r in rr if r["attack"] == "none"]
        worst = lambda m: float(max(np.mean([r[m] for r in att if r["attack"] == x]) for x in ATTACKS[1:]))  # noqa: E731
        e = {"clean_regret": float(np.mean([r["regret"] for r in cl])),
             "clean_opt_frac": float(np.mean([r["opt_frac"] for r in cl])),
             "worst_hijack": worst("hijack"), "worst_hijack_any": worst("hijack_any"),
             "worst_regret": worst("regret")}
        if d.startswith("tab"):
            e.update({"violations": int(sum(not r["cert_sound"] for r in rr)),
                      "action_violations": int(sum(r["action_violations"] for r in rr)),
                      "certified_clean": float(np.mean([r["certified"] for r in cl])),
                      "certified_exact_clean": float(np.mean([r["certified_exact"] for r in cl])),
                      "eps_rel_clean": float(np.median([r["eps_rel"] for r in cl])),
                      "certified_attacked": float(np.mean([r["certified"] for r in att]))})
        summ[f"{rg}|f={f}|rho={rho}|{d}"] = e
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial_offset", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--rhos", default="0.4,0.7")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--defences", default=",".join(DEFS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rhos = [float(x) for x in a.rhos.split(",")]
    defs = a.defences.split(",")
    trials = range(a.trial_offset, a.trial_offset + a.trials)
    if a.smoke:
        trials, a.rounds = range(2), 5
    jobs = [(rg, n, f, rho, at, d, s, a.rounds)
            for (rg, n, f), rho, at, d, s in itertools.product(CELLS, rhos, ATTACKS, defs, trials)]
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "results" / "e08"; out.mkdir(parents=True, exist_ok=True)
    name = "smoke" if a.smoke else f"e08{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = summarise(rows, rhos)
    for kk, v in summ.items():
        print(f"{kk:42} " + " ".join(f"{m}={x:.2f}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = sum(v.get("violations", 0) for v in summ.values())
    aviol = sum(v.get("action_violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "action_violations": aviol,
                                 "seconds": round(time.time() - t0, 1)}))
    log_run("e08_oran", vars(a), {"summary": summ, "violations": viol, "action_violations": aviol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
