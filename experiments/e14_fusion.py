"""E14 - real multi-source tier: test-time memory fed by 36 real web sources (Flight data set, Li et al. 2012).

Each trial is one day: every gold fact (flight, attribute) of that day is one association; its key is a
fixed orthonormal vector and its value is the embedding of the reported 10-minute time bin.  Each of the
36 real sources writes exactly the values it really reported, with its real errors, real coverage and
real copying between sites.  f randomly chosen sources are compromised: on two target facts they report
the attacker's value (gold + 2 h); everywhere else they keep their real reports.

Defences are the E04 code path (`e04.apply_defences`); the decoder reads the memory and returns the
nearest time bin, logits = E y, so the action certificate (11) is exact.  A plurality vote over all
reports (the classical data-fusion baseline, including the compromised sources) is reported as a reference.

Metrics: accuracy on the gold facts (benign = all non-target gold facts of the day), installed / amnesia
on the targets, certified-and-correct fraction, violations, served-installed.

Usage:
    python experiments/e14_fusion.py --smoke
    python experiments/e14_fusion.py --tune_w                      # clean-only eps for tabayyun_w
    python experiments/e14_fusion.py --reps 5 --rep_offset 1500 --tag confirm
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import e04_mqar_attack as e04  # noqa: E402
from tabayyun import Gates  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Unbounded  # noqa: E402
from tabayyun.memory import LinearMemory  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DEFS = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "robustrag", "smsr", "tma_corr",
        "tabayyun", "tabayyun_strict", "tabayyun_w"]
ATTACKS = ["none", "assert", "edge", "unbounded"]
FS = [1, 2, 4]
DK, DV, NBIN = 512, 64, 144
SHIFT = 12                                  # attacker's value: gold + 12 bins (2 hours)
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = json.loads((ROOT / "data" / "fusion" / "flight_facts.json").read_text())
    return _DATA


def embeddings():
    g = torch.Generator().manual_seed(7)
    E = torch.randn(NBIN, DV, generator=g)
    return E / E.norm(dim=1, keepdim=True)


def trial(day, f, attack, rep, defences):
    facts = data()[day]
    F = len(facts)
    g = torch.Generator().manual_seed(zlib.crc32(f"{day}|{f}|{rep}".encode()))   # process-independent seed
    sources = sorted({s for x in facts for s in x["reports"]})
    n = len(sources)
    sidx = {s: i for i, s in enumerate(sources)}
    adv_names = [sources[i] for i in torch.randperm(n, generator=g)[:f].tolist()]
    order = adv_names + [s for s in sources if s not in adv_names]   # adversaries first, as in E04
    pos = {s: i for i, s in enumerate(order)}
    Q, _ = torch.linalg.qr(torch.randn(DK, DK, generator=torch.Generator().manual_seed(11)))
    K = Q[:, :F].T.contiguous()                                      # one orthonormal key per fact
    E = embeddings()
    vals = torch.tensor([x["gold"] for x in facts])
    targets = torch.randperm(F, generator=g)[:2].tolist()
    adv_vals = vals.clone()
    for t in targets:
        adv_vals[t] = (vals[t] + SHIFT) % NBIN
    asserts = [[] for _ in range(n)]
    for j, x in enumerate(facts):
        for s, b in x["reports"].items():
            if s in adv_names and attack != "none" and j in targets:
                continue
            asserts[pos[s]].append((j, b))
    if attack != "none":
        for s in adv_names:
            for t in targets:
                asserts[pos[s]].append((t, int(adv_vals[t])))
    for a in asserts:
        perm = torch.randperm(len(a), generator=g).tolist()
        a[:] = [a[i] for i in perm]
    # per-source quarantines: one Titans write per report (orthonormal keys, theta = 0.5 stores exactly)
    Qm = LinearMemory(n, DK, DV, Gates(theta=0.5, eta=0.0, alpha=0.0))
    written = []
    for t in range(max(len(a) for a in asserts)):
        k = torch.zeros(n, DK); v = torch.zeros(n, DV); mask = torch.zeros(n, dtype=torch.bool)
        for s, a in enumerate(asserts):
            if t < len(a):
                j, b = a[t]
                k[s], v[s], mask[s] = K[j], E[b], True
        Qm.write(k, v, mask)
    written = [(s, K[j]) for s, a in enumerate(asserts) for j in sorted({j for j, _ in a})]

    def titans():
        flat = [(j, b) for a in asserts for (j, b) in a]
        T = LinearMemory(1, DK, DV, Gates(theta=0.5, eta=0.5, alpha=0.005))
        for i in torch.randperm(len(flat), generator=g).tolist():
            j, b = flat[i]
            T.write(K[j][None], E[b][None])
        return T.M[0]

    hook = None
    if attack in ("edge", "unbounded"):
        ks, vs = K[targets[0]], E[int(adv_vals[targets[0]])]
        hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                else Unbounded(list(range(f)), ks, vs, scale=20.0))
    benign = [j for j in range(F) if j not in targets]
    res = e04.apply_defences(defences, Qm.M, written, K, titans, lambda y: y @ E.T, lambda: E, hook, n, f, F,
                             rep, "cpu", vals, adv_vals, targets, benign, attack)
    # classical data fusion: plurality vote over every report (compromised sources included)
    vote = []
    for j in range(F):
        c = Counter(b for a in asserts for (jj, b) in a if jj == j)
        vote.append(c.most_common(1)[0][0] if c else -1)
    vote = torch.tensor(vote)
    res["vote"] = {"recall": float((vote[benign] == vals[benign]).float().mean()),
                   "installed": float(np.mean([int(vote[t] == adv_vals[t]) for t in targets])) if attack != "none" else None,
                   "amnesia": float(np.mean([int(vote[t] != vals[t]) for t in targets]))}
    return res


def job(j):
    day, f, attack, rep, eps, defences = j
    torch.set_num_threads(1)
    if eps is not None:
        e04.EPS_W = eps
    out = trial(day, f, attack, rep, defences)
    return [{"day": day, "f": f, "attack": attack, "trial": rep, "defence": d, **r} for d, r in out.items()]


def summarise(rows, defs):
    S = {}
    for f, d in itertools.product(FS, defs + ["vote"]):
        rr = [r for r in rows if r["f"] == f and r["defence"] == d]
        if not rr:
            continue
        cl = [r for r in rr if r["attack"] == "none"]
        at = [r for r in rr if r["attack"] != "none"]
        e = {"clean_accuracy": float(np.mean([r["recall"] for r in cl])),
             "worst_installed": float(max(np.mean([r["installed"] for r in at if r["attack"] == a]) for a in ATTACKS[1:])),
             "worst_amnesia": float(max(np.mean([r["amnesia"] for r in at if r["attack"] == a]) for a in ATTACKS[1:]))}
        if d in e04.CERT_METHODS:
            e.update({"violations": int(sum(r["violations"] + r["action_violations"] for r in rr)),
                      "certified_correct_clean": float(np.mean([r["certified_correct_frac"] for r in cl])),
                      "worst_served_installed": float(max(np.mean([r["served_installed"] for r in at if r["attack"] == a])
                                                          for a in ATTACKS[1:]))})
        S[f"f={f}|{d}"] = e
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rep_offset", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune_w", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "e14"; out.mkdir(parents=True, exist_ok=True)
    days = sorted(data())
    tf = out / "tuning_w_clean.json"
    if a.tune_w:                      # clean-only, exploratory reps 0-1, same rule as the other tiers
        grid, res = (0.1, 0.2, 0.3, 0.4, 0.6), {}
        for eps in grid:
            jobs = [(d, f, "none", r, eps, ["tabayyun_w"]) for d, f, r in itertools.product(days, FS, range(2))]
            with ProcessPoolExecutor(a.workers) as ex:
                rows = [x for rr in ex.map(job, jobs) for x in rr if x["defence"] == "tabayyun_w"]
            res[str(eps)] = float(np.mean([r["recall"] for r in rows]) + np.mean([r["certified_correct_frac"] for r in rows]))
            print(f"eps={eps} objective={res[str(eps)]:.4f}", flush=True)
        best = max(res.values())
        res["selected_eps"] = min(e for e in grid if res[str(e)] >= best - 0.02)
        tf.write_text(json.dumps(res, indent=1))
        print("selected eps", res["selected_eps"])
        return
    eps = json.loads(tf.read_text())["selected_eps"] if tf.exists() else None
    reps = range(a.rep_offset, a.rep_offset + a.reps)
    days_used = days
    if a.smoke:
        days_used, reps = days[:3], range(1)
    jobs = [(d, f, at, r, eps, DEFS) for d, f, at, r in itertools.product(days_used, FS, ATTACKS, reps)]
    print(f"{len(jobs)} trials on {a.workers} workers, eps_w={eps if eps is not None else e04.EPS_W}", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, rr in enumerate(ex.map(job, jobs)):
            rows.extend(rr)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    name = "smoke" if a.smoke else f"e14{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    S = summarise(rows, DEFS)
    for k, v in S.items():
        print(f"{k:22} " + " ".join(f"{m}={x:.3f}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()), flush=True)
    (out / f"{name}_summary.json").write_text(json.dumps(S, indent=1))
    viol = sum(v.get("violations", 0) for v in S.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "seconds": round(time.time() - t0, 1)}))
    log_run("e14_fusion", vars(a), {"summary": S, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
