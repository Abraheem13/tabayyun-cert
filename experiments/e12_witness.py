"""E12 - witness complexity (Theorem 6): how many honest sources must know a fact before a
defence serves it, and what happens to that number as the population n grows.

World (Tier A): n sources, the first f adversarial.  Every fact j is known to EXACTLY m_j
honest sources (m_j on a grid from 1 to n - f); its witnesses re-assert it in every round
(rho = 1), so the only question is the number of witnesses.

  clean run  : per fact, is it served?  (||read - v|| < 0.5 ||v||)  -> recall as a function of m
  attack run : the adversaries attack ONE fact with m witnesses (v* = -v):
               installed / flip / amnesia as a function of m

Predictions (Theorem 6, Amendment 14):
  hull-valid Tabayyun (cone gate, k = 2f+1) serves a fact iff m >= 2f+1, for every n;
  witness-optimal Tabayyun-W (agreement gate, k = f+1) serves it iff m >= f+1, for every n;
  majority aggregators need m ~ n/2, which grows with n;
  rules that serve with fewer than f+1 witnesses, or count witnesses without checking
  agreement, are forgeable.

Usage: python experiments/e12_witness.py --seeds 10 --seed_offset 1000
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
from tabayyun.data.streams import FactWorld, WorldConfig, run_round  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("e02", ROOT / "experiments" / "e02_synthetic_defence.py")
e02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e02)

DEFS = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_geomed", "robustrag", "tma_corr",
        "tabayyun_anchored2", "tabayyun_w"]
TAB = ("tabayyun_anchored2", "tabayyun_w")
ATTACKS = ["targeted", "edge_of_range", "agree_edge", "unbounded"]
CONFIGS = [(10, 2), (20, 2), (40, 2), (80, 2), (40, 1), (40, 3)]            # (n, f)
ROUNDS = 6


def grid(n, f):
    h = n - f
    g = {1, f, f + 1, 2 * f, 2 * f + 1, 2 * f + 2, round(n / 4), round(n / 2) - 1, round(n / 2) + 1,
         round(3 * n / 4), h}
    return sorted(m for m in g if 1 <= m <= h)


def make_world(n, f, seed):
    ms = grid(n, f)
    wc = WorldConfig(n_sources=n, n_adv=f, dk=32, dv=32, n_facts=len(ms), rho=1.0, sigma=0.1, tau=50, seed=seed)
    w = FactWorld(wc)
    g = torch.Generator().manual_seed(seed + 7)
    w.obs_p.zero_()
    for j, m in enumerate(ms):
        wit = w.honest[torch.randperm(len(w.honest), generator=g)[:m]]
        w.obs_p[wit, j] = 1.0
        w.support[j] = m
    return wc, w, ms


def one(job):
    n, f, defence, attack, seed, eps = job[:6]
    target = job[6] if len(job) > 6 else None
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    e02.EPS_W = eps
    wc, w, ms = make_world(n, f, seed)
    mem = e02.build(defence, wc, f, 2 * f + 1, "cpu")
    tab = defence in TAB
    if tab:
        mem.oracle_honest = w.honest
    watk = None
    if attack != "none":
        k_star, v_true = w.keys[target], w.values[target]
        v_star = -v_true
        watk, hook = e02.make_attack(attack, w, mem, k_star, v_star)
        if hook is not None:
            mem.delta_hook = hook
    sound, ratio = True, 0.0
    for _ in range(ROUNDS):
        run_round(mem, w, watk)
        if tab:
            td, ce = mem.true_drift(), mem.certificate
            sound &= td <= ce * (1 + 1e-4) + 1e-5
            if td > 1e-6:
                ratio = max(ratio, td / max(ce, 1e-12))
    Y = mem.read(w.keys)
    base = {"n": n, "f": f, "defence": defence, "attack": attack, "seed": seed}
    if tab:
        base.update({"cert_sound": bool(sound), "drift_over_cert": ratio})
    if attack == "none":
        ok = ((Y - w.values).norm(dim=-1) < 0.5 * w.values.norm(dim=-1)).float()
        return [{**base, "m": m, "recall": float(ok[j])} for j, m in enumerate(ms)]
    y = Y[target]
    return [{**base, "m": ms[target],
             "installed": float((y - v_star).norm() < 0.5 * v_star.norm()),
             # a wrong answer is SERVED: the read is not negligible and is closer to v* than to v.
             # (For an unserved fact the read is ~0 and "closer to v*" is a floating-point coin toss.)
             "flip": float(y.norm() >= 0.1 * v_true.norm() and (y - v_star).norm() < (y - v_true).norm()),
             "abstained": float(y.norm() < 0.1 * v_true.norm()),
             "amnesia": float((y - v_true).norm() >= 0.5 * v_true.norm()),
             "shift": float((y - v_true).norm())}]


def threshold(curve):
    """Smallest grid m from which the fact is served (recall >= 0.5 there and at every larger m)."""
    ms = sorted(curve)
    for i, m in enumerate(ms):
        if all(curve[x] >= 0.5 for x in ms[i:]):
            return m
    return None


def summarise(rows):
    S = {}
    for (n, f), d in itertools.product(CONFIGS, DEFS):
        rr = [r for r in rows if (r["n"], r["f"], r["defence"]) == (n, f, d)]
        if not rr:
            continue
        ms = sorted({r["m"] for r in rr})
        cl = {m: float(np.mean([r["recall"] for r in rr if r["attack"] == "none" and r["m"] == m])) for m in ms}
        att = [r for r in rr if r["attack"] != "none"]
        cell = lambda metric: {m: float(max(np.mean([r[metric] for r in att if r["m"] == m and r["attack"] == a])  # noqa: E731
                                            for a in sorted({r["attack"] for r in att}))) for m in ms}
        e = {"witness_threshold": threshold(cl), "threshold_over_n": None, "recall_by_m": cl}
        if e["witness_threshold"] is not None:
            e["threshold_over_n"] = round(e["witness_threshold"] / n, 3)
        if att:
            inst, flip, amn = cell("installed"), cell("flip"), cell("amnesia")
            e.update({"worst_installed": max(inst.values()), "worst_flip": max(flip.values()),
                      "worst_amnesia_served": max((amn[m] for m in ms if cl[m] >= 0.5), default=0.0),
                      "installed_by_m": inst, "flip_by_m": flip, "amnesia_by_m": amn})
        if d in TAB:
            e["violations"] = int(sum(not r["cert_sound"] for r in rr))
            e["max_drift_over_cert"] = float(max(r["drift_over_cert"] for r in rr))
        S[f"n={n}|f={f}|{d}"] = e
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=None, help="agreement tolerance (default: clean-tuned value)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "e12"; out.mkdir(parents=True, exist_ok=True)
    eps = a.eps
    tf = out / "tuning_clean.json"
    if eps is None:
        eps = json.loads(tf.read_text())["selected_eps"] if tf.exists() else e02.EPS_W
    configs, seeds = CONFIGS, range(a.seed_offset, a.seed_offset + a.seeds)
    if a.smoke:
        configs, seeds = [(10, 2), (40, 2)], range(2)
    jobs = []
    for (n, f), d, s in itertools.product(configs, DEFS, seeds):
        jobs.append((n, f, d, "none", s, eps))
        for at, t in itertools.product(ATTACKS, range(len(grid(n, f)))):
            jobs.append((n, f, d, at, s, eps, t))
    jobs.sort(key=lambda j: -j[0])                      # big populations first (better load balance)
    print(f"{len(jobs)} runs on {a.workers} workers, eps={eps}", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, rr in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.extend(rr)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    name = "smoke" if a.smoke else f"e12{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    S = summarise([r for r in rows if (r["n"], r["f"]) in configs])
    for k, v in S.items():
        print(f"{k:34} threshold m*={v['witness_threshold']} (m*/n={v['threshold_over_n']}) "
              f"worst_installed={v.get('worst_installed', float('nan')):.2f} worst_flip={v.get('worst_flip', float('nan')):.2f} "
              f"amnesia_when_served={v.get('worst_amnesia_served', float('nan')):.2f}"
              + (f" violations={v['violations']} drift/cert={v['max_drift_over_cert']:.2f}" if "violations" in v else ""))
    (out / f"{name}_summary.json").write_text(json.dumps(S, indent=1))
    viol = sum(v.get("violations", 0) for v in S.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "eps": eps, "seconds": round(time.time() - t0, 1)}))
    log_run("e12_witness", {**vars(a), "eps_used": eps}, {"summary": S, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
