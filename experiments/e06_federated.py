"""E06 - federated two-level Tabayyun (Tier A memories), Byzantine clients + local adversaries.

World: F global facts (orthonormal keys).  N clients; each client has n sources,
the first f of which are adversarial inside every client.  Locality across
clients: each fact is relevant at m ~ U[k_c, N - f_c] honest clients (dense:
all honest clients).  Inside a relevant client every honest source observes
the fact w.p. rho per round (with a per-source bias sigma).  The first f_c
clients are Byzantine at the server (they send crafted rows and under-report
their local bound).

Defences (all with the same local + server adversaries):
  fed_tabayyun   local Tabayyun + server tabayyun gate  (Thm 5 certificate)
  fed_mean       local Tabayyun + server plain mean (FedAvg of promoted memories)
  fed_trmean / fed_median / fed_krum / fed_geomed   local Tabayyun + server robust aggregator
  fedavg_titans  local ungated Titans + server FedAvg (the FedNL-style substrate)
Metrics: global flip ASR at the attacked key (L2), installed, Recall_L2 over facts,
certificate soundness (true drift vs Thm-5 certificate).

Usage: python experiments/e06_federated.py --trials 20 --trial_offset 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun import GateConfig, Gates, TabayyunConfig, TabayyunMemory  # noqa: E402
from tabayyun.attacks import EdgeOfRange, TargetedWrite, Unbounded  # noqa: E402
from tabayyun.baselines import SharedMemory  # noqa: E402
from tabayyun.federated import FederatedTabayyun  # noqa: E402
from tabayyun.gate import AGGREGATORS  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DEFS = ["fed_tabayyun", "fed_mean", "fed_trmean", "fed_median", "fed_krum", "fed_geomed", "fedavg_titans"]
ATTACKS = ["none", "local", "server_edge", "server_unbounded", "both"]


class FedWorld:
    def __init__(self, N, n, f, f_c, F, dk, dv, rho, sigma, regime, seed, tau=50, reps=2):
        g = torch.Generator().manual_seed(seed)
        self.g, self.N, self.n, self.f, self.f_c, self.F, self.tau, self.reps = g, N, n, f, f_c, F, tau, reps
        Q, _ = torch.linalg.qr(torch.randn(dk, dk, generator=g))
        self.keys = Q[:, :F].T.contiguous()
        v = torch.randn(F, dv, generator=g)
        self.values = v / v.norm(dim=1, keepdim=True)
        self.bias = sigma * torch.randn(N, n, F, dv, generator=g) / dv ** 0.5
        honest_clients = list(range(f_c, N))
        kc = 2 * f_c + 1
        self.relevant = torch.zeros(N, F, dtype=torch.bool)
        for j in range(F):
            if regime == "dense":
                cl = honest_clients
            else:
                m = int(torch.randint(kc, len(honest_clients) + 1, (1,), generator=g))
                cl = [honest_clients[i] for i in torch.randperm(len(honest_clients), generator=g)[:m]]
            self.relevant[cl, j] = True
        self.relevant[:f_c] = True        # Byzantine clients may "know" anything
        self.rho = rho

    def client_schedule(self, c):
        sched = [[] for _ in range(self.n)]
        for s in range(self.f, self.n):
            obs = (torch.rand(self.F, generator=self.g) < self.rho) & self.relevant[c]
            fs = torch.nonzero(obs).flatten().tolist() * self.reps
            perm = torch.randperm(len(fs), generator=self.g).tolist()
            sched[s] = [fs[i] for i in perm][: self.tau]
        return sched


def one(job):
    regime, f_c, rho, attack, defence, seed, rounds = job
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    N, n, f, F, dk, dv = 12, 7, 1, 16, 32, 32
    w = FedWorld(N, n, f, f_c, F, dk, dv, rho, 0.1, regime, seed)
    k_local, k_srv = 2 * f + 1, 2 * f_c + 1
    target = 0
    k_star, v_true = w.keys[target], w.values[target]
    v_star = -v_true
    gates = Gates(theta=0.5, alpha=0.005)
    titans = defence == "fedavg_titans"
    clients = []
    for c in range(N):
        if titans:
            clients.append(SharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=0.005)))
        else:
            m = TabayyunMemory(TabayyunConfig(n_sources=n, dk=dk, dv=dv, tau=w.tau, q_gates=gates,
                                              gate=GateConfig(k=k_local, f=f, gamma=0.3, nu=0.05),
                                              basis="anchored", anchor_min_writers="f+1"))
            m.oracle_honest = torch.arange(f, n)
            if attack in ("local", "both"):
                m.delta_hook = EdgeOfRange(list(range(f)), list(range(f, n)), k_star, v_star, nu=0.05)
            clients.append(m)
    local_writer = TargetedWrite(list(range(f)), k_star, v_star) if attack in ("local", "both") else None
    fed = None
    if not titans:
        fed = FederatedTabayyun(clients, GateConfig(k=k_srv, f=f_c, gamma=0.3, nu=0.05,
                                                    corroborate=(defence == "fed_tabayyun")), alpha_G=0.5)
        fed.byzantine = list(range(f_c))
        if attack in ("server_edge", "both"):
            fed.server_hook = EdgeOfRange(list(range(f_c)), list(range(f_c, N)), k_star, v_star, nu=0.05)
        elif attack == "server_unbounded":
            fed.server_hook = Unbounded(list(range(f_c)), k_star, v_star, scale=50.0)
    MG = torch.zeros(dv, dk)
    sound = True
    for _ in range(rounds):
        for c, cl in enumerate(clients):
            sched = w.client_schedule(c)
            for t in range(w.tau):
                src, ks_, vs_ = [], [], []
                for s in range(f, n):
                    if t < len(sched[s]):
                        j = sched[s][t]
                        src.append(s); ks_.append(w.keys[j]); vs_.append(w.values[j] + w.bias[c, s, j])
                if local_writer is not None and c >= f_c:
                    for s, kk, vv in local_writer.writes(t, cl):
                        src.append(s); ks_.append(kk); vs_.append(vv)
                if src:
                    cl.write(torch.tensor(src), torch.stack(ks_), torch.stack(vs_))
                elif hasattr(cl, "promote"):
                    cl.write(torch.zeros(0, dtype=torch.long), torch.zeros(0, dk), torch.zeros(0, dv))
        if defence == "fed_tabayyun":
            fed.sync()
            sound &= fed.true_drift() <= fed.certificate * (1 + 1e-4) + 1e-5
        else:
            # server aggregation of client memories (identity basis), Byzantine rows crafted the same way
            I = torch.eye(dk)
            if titans:
                D = torch.stack([I @ cl.mem.M[0].T for cl in clients])
            else:
                D = torch.stack([I @ cl.MP.T for cl in clients])
            if attack in ("server_edge", "both"):
                D = EdgeOfRange(list(range(f_c)), list(range(f_c, N)), k_star, v_star, nu=0.05)(D, I)
            elif attack == "server_unbounded":
                D = Unbounded(list(range(f_c)), k_star, v_star, scale=50.0)(D, I)
            agg = "mean" if defence in ("fed_mean", "fedavg_titans") else defence[4:]
            val = AGGREGATORS[agg](D, f_c)
            MG = MG + 0.5 * (val - I @ MG.T).T
    read = (lambda q: fed.read(q)) if defence == "fed_tabayyun" else (lambda q: q @ MG.T)
    y = read(k_star[None])[0]
    others = torch.arange(1, F)
    Y = read(w.keys[others])
    rec = float(((Y - w.values[others]).norm(dim=-1) < 0.5).float().mean())
    r = {"regime": regime, "f_c": f_c, "rho": rho, "attack": attack, "defence": defence, "seed": seed,
         "recall_l2": rec,
         "flip": float((y - v_star).norm() < (y - v_true).norm()) if attack != "none" else None,
         "installed": float((y - v_star).norm() < 0.5) if attack != "none" else None,
         "amnesia": float((y - v_true).norm() >= 0.5)}
    if defence == "fed_tabayyun":
        r.update({"cert_sound": bool(sound), "cert": fed.certificate, "true_drift": fed.true_drift(),
                  "n_global_dirs": int(fed.U.shape[0])})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial_offset", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cells = [(rg, fc, rho) for rg in ("dense", "locality") for fc in (1, 2) for rho in (0.4, 0.7)]
    trials = range(a.trial_offset, a.trial_offset + a.trials)
    if a.smoke:
        cells, trials, a.rounds = [("dense", 1, 0.7), ("locality", 2, 0.7)], range(2), 4
    jobs = [(rg, fc, rho, at, d, s, a.rounds) for (rg, fc, rho), at, d, s in
            itertools.product(cells, ATTACKS, DEFS, trials)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    out = ROOT / "results" / "e06"; out.mkdir(parents=True, exist_ok=True)
    name = "smoke" if a.smoke else f"e06{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    for (rg, fc, rho) in cells:
        for d in DEFS:
            rr = [r for r in rows if (r["regime"], r["f_c"], r["rho"], r["defence"]) == (rg, fc, rho, d)]
            att = [r for r in rr if r["attack"] != "none"]
            worst = lambda m: max(np.mean([r[m] for r in att if r["attack"] == x]) for x in ATTACKS[1:])  # noqa: E731
            e = {"clean_recall": float(np.mean([r["recall_l2"] for r in rr if r["attack"] == "none"])),
                 "worst_flip": float(worst("flip")), "worst_installed": float(worst("installed")),
                 "worst_amnesia": float(worst("amnesia"))}
            if d == "fed_tabayyun":
                e["violations"] = int(sum(not r["cert_sound"] for r in rr))
            summ[f"{rg}|fc={fc}|rho={rho}|{d}"] = e
            print(f"{rg:8} fc={fc} rho={rho} {d:14} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = sum(v.get("violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "seconds": round(time.time() - t0, 1)}))
    log_run("e06_federated", vars(a), {"summary": summ, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
