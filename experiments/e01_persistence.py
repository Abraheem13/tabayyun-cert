"""E01 - Hypotheses 1 and 5 on ungated Titans memory.

Part `halflife` (H1): how long does one poisoned write persist?
    Measured with a counterfactual twin (poisoned vs clean memory on the same
    stream).  Compared against
      * Lemma 1 (exact for an orthogonal / quiet stream),
      * Lemma 1' mean-field law (random stream; benign writes self-correct
        at rate 2 theta / d_k, amplified by momentum),
      * the v1 formula ln2 / -ln max(eta, 1-alpha) (kept for the record).
Part `rayn` (H5 / Prop. 0): repeated low-norm writes vs one large write under
    per-write norm filters (clip G).

Usage:
    python experiments/e01_persistence.py --smoke
    python experiments/e01_persistence.py --device cuda --part halflife --kinds mlp --stream random
    python experiments/e01_persistence.py --device cuda --part rayn
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabayyun.certify import half_life, mean_field_half_life  # noqa: E402
from tabayyun.memory import Gates, make_memory  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DIVERGE = 1e6


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def benign_key(g, dk, device, k_star, stream):
    k = torch.randn(1, dk, generator=g).to(device)
    if stream == "orthogonal":
        k = k - (k @ k_star.T) * k_star
    return unit(k)


def measure_half_life(kind, dk, dv, eta, alpha, theta, T, seed, device, stream="random"):
    """Counterfactual twin.  Returns (half_life | inf | None if diverged, traj)."""
    g = torch.Generator().manual_seed(seed)
    gates = Gates(theta=theta, eta=eta, alpha=alpha)
    mp = make_memory(kind, 1, dk, dv, gates, device=device)
    mc = make_memory(kind, 1, dk, dv, gates, device=device)
    k_star = unit(torch.randn(1, dk, generator=g)).to(device)
    v_star = unit(torch.randn(1, dv, generator=g)).to(device)
    mp.write(k_star, v_star)
    mc.write(k_star, mc.read(k_star))                 # clean twin: zero-gradient write, same clock
    traj = []
    for _ in range(T):
        k = benign_key(g, dk, device, k_star, stream)
        v = unit(torch.randn(1, dv, generator=g)).to(device)
        mp.write(k, v); mc.write(k, v)
        x = float((mp.read(k_star) - mc.read(k_star)).norm())
        if not math.isfinite(x) or x > DIVERGE:
            return None, traj
        traj.append(x)
    peak = max(traj)
    pk = traj.index(peak)
    for t in range(pk, T):
        if traj[t] <= 0.5 * peak:
            return t - pk, traj
    return math.inf, traj


def attack_run(name, dk, dv, eta, alpha, theta, clip, n_attack, seed, device, prefill=60, refresh=25):
    """Flip an *established* honest fact (k*, v_true) into (k*, v*).

    The fact is first learned from `prefill` honest writes and is refreshed by
    one honest write every `refresh` steps (it keeps being observed).  The
    attack (single | rayn) is interleaved 1:1 with benign random writes.
    Success = the read at k* projects more on v* than on v_true (a flip).
    Per-write filter: Gates.clip bounds the gradient norm ||g|| <= G.
    """
    g = torch.Generator().manual_seed(seed)
    m = make_memory("linear", 1, dk, dv, Gates(theta=theta, eta=eta, alpha=alpha, clip=clip), device=device)
    k_star = unit(torch.randn(1, dk, generator=g)).to(device)
    v_true = unit(torch.randn(1, dv, generator=g)).to(device)
    v_star = unit(torch.randn(1, dv, generator=g)).to(device)
    v_star = unit(v_star - (v_star @ v_true.T) * v_true)          # orthogonal target
    for _ in range(prefill):
        m.write(k_star, v_true)
    # gradient norm of a rayn write is 2*eps (unit key), keep it under the filter
    eps = 0.1 if clip is None else min(0.1, 0.9 * clip / 2)
    norms, hit, flipped_end = [], math.inf, False
    peak_margin = -math.inf
    for t in range(n_attack):
        if name == "single":
            if t == 0:
                norms.append(float(m.write(k_star, 10.0 * v_star)[0]))
        else:
            cur = m.read(k_star)
            d = v_star - cur
            norms.append(float(m.write(k_star, cur + d * min(1.0, eps / (float(d.norm()) + 1e-12)))[0]))
        m.write(unit(torch.randn(1, dk, generator=g)).to(device), unit(torch.randn(1, dv, generator=g)).to(device))
        if (t + 1) % refresh == 0:
            m.write(k_star, v_true)
        y = m.read(k_star)
        margin = float(y @ v_star.T - y @ v_true.T)
        peak_margin = max(peak_margin, margin)
        if margin > 0 and hit == math.inf:
            hit = t + 1
    flipped_end = margin > 0
    return {"flipped_end": flipped_end, "flipped_ever": hit < math.inf, "writes_to_flip": hit,
            "final_margin": margin, "peak_margin": peak_margin, "max_write_norm": max(norms)}


def part_halflife(a, kinds, streams, etas, alphas, seeds, T, theta, dk, dv):
    rows = []
    for kind, stream, eta, alpha in itertools.product(kinds, streams, etas, alphas):
        res = [measure_half_life(kind, dk, dv, eta, alpha, theta, T, s, a.device, stream)[0] for s in seeds]
        ok = [h for h in res if h is not None]
        lam = 0.0 if stream == "orthogonal" else 2 * theta / dk
        rows.append({"kind": kind, "stream": stream, "eta": eta, "alpha": alpha, "theta": theta,
                     "n_diverged": len(res) - len(ok),
                     "half_life_measured": (sum(ok) / len(ok)) if ok else None,
                     "half_life_seeds": ok,
                     "pred_lemma1": mean_field_half_life(eta, alpha, 0.0),
                     "pred_meanfield": mean_field_half_life(eta, alpha, lam),
                     "pred_v1": half_life(eta, alpha)})
        print(json.dumps(rows[-1]))
    return rows


def fit(rows, key):
    pts = [(r[key], r["half_life_measured"]) for r in rows
           if r["half_life_measured"] not in (None, math.inf) and r["half_life_measured"] > 0
           and math.isfinite(r[key]) and r[key] > 0]
    if len(pts) < 3:
        return {"n": len(pts)}
    p, m = map(np.array, zip(*pts))
    lr = np.log(m / p)
    return {"n": len(pts), "loglog_corr": float(np.corrcoef(np.log(p), np.log(m))[0, 1]),
            "median_ratio": float(np.exp(np.median(lr))), "max_abs_log_ratio": float(np.abs(lr).max()),
            "frac_measured_le_pred": float(np.mean(m <= p * 1.02))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--part", default="all", choices=["all", "halflife", "rayn"])
    ap.add_argument("--kinds", default="linear,mlp")
    ap.add_argument("--streams", default="random,orthogonal")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/e01")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    sm = a.smoke
    seeds = range(2 if sm else a.seeds)
    dk = dv = 64
    theta = 0.05
    out, metrics = {}, {}
    if a.part in ("all", "halflife"):
        rows = part_halflife(a, ["linear"] if sm else a.kinds.split(","),
                             ["random"] if sm else a.streams.split(","),
                             [0.0, 0.5, 0.9] if sm else [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                             [0.01, 0.05] if sm else [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                             seeds, 400 if sm else 3000, theta, dk, dv)
        out["half_life"] = rows
        for kind in {r["kind"] for r in rows}:
            for stream in {r["stream"] for r in rows}:
                rr = [r for r in rows if r["kind"] == kind and r["stream"] == stream]
                if not rr:
                    continue
                key = f"{kind}_{stream}"
                metrics[f"{key}_vs_meanfield"] = fit(rr, "pred_meanfield")
                metrics[f"{key}_vs_lemma1"] = fit(rr, "pred_lemma1")
                metrics[f"{key}_vs_v1"] = fit(rr, "pred_v1")
                metrics[f"{key}_diverged_runs"] = sum(r["n_diverged"] for r in rr)
    if a.part in ("all", "rayn"):
        clips = [None, 0.05] if sm else [None, 0.5, 0.2, 0.1, 0.05, 0.02]
        rows = []
        for clip, eta, s in itertools.product(clips, [0.0, 0.9], seeds):
            for name in ["single", "rayn"]:
                r = attack_run(name, dk, dv, eta, 0.01, theta, clip, 150 if sm else 600, s, a.device)
                r.update({"attack": name, "clip": clip, "eta": eta, "seed": s})
                rows.append(r)
        out["rayn"] = rows
        summ = {}
        for clip, name in itertools.product(clips, ["single", "rayn"]):
            rr = [r for r in rows if r["clip"] == clip and r["attack"] == name]
            hits = [r["writes_to_flip"] for r in rr if r["writes_to_flip"] < math.inf]
            summ[f"{name}@clip={clip}"] = {
                "asr_end": float(np.mean([r["flipped_end"] for r in rr])),
                "asr_ever": float(np.mean([r["flipped_ever"] for r in rr])),
                "median_writes_to_flip": float(np.median(hits)) if hits else None,
                "max_write_norm": float(np.max([r["max_write_norm"] for r in rr]))}
            print(f"{name:6} clip={clip}: {summ[f'{name}@clip={clip}']}")
        metrics["rayn"] = summ
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fname = "smoke.json" if sm else f"e01_{a.part}{('_' + a.tag) if a.tag else ''}.json"
    (Path(a.out) / fname).write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps(metrics))
    log_run("e01_persistence", vars(a), metrics, out="results/smoke.jsonl" if sm else "results/registry.jsonl",
            smoke=sm)


if __name__ == "__main__":
    main()
