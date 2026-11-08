"""E01 - Hypothesis 1 (persistence law + rayn attack) on ungated Titans memory.

H1a: the half-life of a poisoned association scales as ln2 / -ln max(eta, 1-alpha).
H1b: a repeated low-norm 'rayn' attack reaches the same targeted ASR as one
     large write at a fraction of the per-write norm, so it evades per-write
     norm filters.

Usage:
    python experiments/e01_persistence.py --smoke          # 1 min, CPU
    python experiments/e01_persistence.py --device cuda    # full grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabayyun.certify import half_life  # noqa: E402
from tabayyun.memory import Gates, make_memory  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402


def exact_half_life(eta, alpha, T=100000):
    """Half-life of the exact single-write curve sum_j (1-a)^{T-j} eta^j (Lemma 1),
    measured from its peak.  The v1 formula ln2/-ln max(eta,1-alpha) is only
    its asymptotic rate and ignores the momentum amplification."""
    a = 1.0 - alpha
    vals, s_prev = [], 0.0
    for t in range(T):
        s_prev = a * s_prev + eta ** t
        vals.append(s_prev)
        if t > 10 and vals[-1] < vals[-2] and vals[-1] <= 0.5 * max(vals):
            pk = vals.index(max(vals))
            return t - pk
    return math.inf


def measure_half_life(kind, dk, dv, eta, alpha, theta, T, seed, device):
    """Counterfactual twin: identical benign streams with and without one
    poisoned write (k*, v*).  The poison's causal footprint is
    ||M_poison(k*) - M_clean(k*)||; return steps from its peak to half-peak."""
    g = torch.Generator().manual_seed(seed)
    gates = Gates(theta=theta, eta=eta, alpha=alpha)
    mp = make_memory(kind, 1, dk, dv, gates, device=device)
    mc = make_memory(kind, 1, dk, dv, gates, device=device)
    k_star = torch.nn.functional.normalize(torch.randn(1, dk, generator=g), dim=1).to(device)
    v_star = torch.randn(1, dv, generator=g).to(device)
    mp.write(k_star, v_star)
    mc.write(k_star, mc.read(k_star))                 # clean twin: zero-gradient write, same clock
    traj = []
    for _ in range(T):
        k = torch.nn.functional.normalize(torch.randn(1, dk, generator=g), dim=1).to(device)
        v = torch.randn(1, dv, generator=g).to(device)
        mp.write(k, v); mc.write(k, v)
        traj.append(float((mp.read(k_star) - mc.read(k_star)).norm()))
    peak = max(traj)
    pk = traj.index(peak)
    for t in range(pk, T):
        if traj[t] <= 0.5 * peak:
            return t - pk, traj
    return math.inf, traj


def rayn_vs_single(dk, dv, eta, alpha, theta, n_writes, seed, device, filt=None):
    """Return (ASR proxy = cos(M k*, v*), max per-write surprise norm) for both attacks."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    k_star = torch.nn.functional.normalize(torch.randn(1, dk, generator=g), dim=1).to(device)
    v_star = torch.randn(1, dv, generator=g).to(device)
    v_star = v_star / v_star.norm()
    for name in ["single", "rayn"]:
        m = make_memory("linear", 1, dk, dv, Gates(theta=theta, eta=eta, alpha=alpha, clip=filt), device=device)
        norms = []
        for t in range(n_writes):
            if name == "single":
                if t == 0:
                    norms.append(float(m.write(k_star, v_star)[0]))
                k = torch.nn.functional.normalize(torch.randn(1, dk, generator=g), dim=1).to(device)
                m.write(k, torch.randn(1, dv, generator=g).to(device) / dv ** 0.5)
            else:
                cur = m.read(k_star)
                d = v_star - cur
                step = d * min(1.0, 0.1 / (float(d.norm()) + 1e-12))
                norms.append(float(m.write(k_star, cur + step)[0]))
                k = torch.nn.functional.normalize(torch.randn(1, dk, generator=g), dim=1).to(device)
                m.write(k, torch.randn(1, dv, generator=g).to(device) / dv ** 0.5)
        cs = float(torch.nn.functional.cosine_similarity(m.read(k_star), v_star, dim=1))
        out[name] = {"cos_target": cs, "max_write_norm": max(norms), "n_attack_writes": len(norms)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/e01")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    etas = [0.0, 0.5, 0.9] if a.smoke else [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    alphas = [0.01, 0.05] if a.smoke else [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    seeds = range(2 if a.smoke else a.seeds)
    kinds = ["linear"] if a.smoke else ["linear", "mlp"]
    T = 400 if a.smoke else 3000
    dk = dv = 64
    rows = []
    for kind, eta, alpha in itertools.product(kinds, etas, alphas):
        hls = [measure_half_life(kind, dk, dv, eta, alpha, 0.05, T, s, a.device)[0] for s in seeds]
        rows.append({"kind": kind, "eta": eta, "alpha": alpha,
                     "half_life_measured": sum(hls) / len(hls),
                     "half_life_pred_v1_asymptotic": half_life(eta, alpha),
                     "half_life_pred": exact_half_life(eta, alpha)})
        print(json.dumps(rows[-1]))
    rayn = [rayn_vs_single(dk, dv, 0.9, 0.01, 0.05, 60 if a.smoke else 200, s, a.device) for s in seeds]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / ("smoke.json" if a.smoke else "e01.json")).write_text(
        json.dumps({"half_life": rows, "rayn": rayn}, indent=1))
    finite = [r for r in rows if math.isfinite(r["half_life_measured"]) and math.isfinite(r["half_life_pred"])]
    if len(finite) > 2:
        import numpy as np
        x = np.log([r["half_life_pred"] for r in finite]); y = np.log([r["half_life_measured"] for r in finite])
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = float("nan")
    metrics = {"loglog_corr_pred_vs_measured": corr,
               "rayn_cos_target_mean": sum(r["rayn"]["cos_target"] for r in rayn) / len(rayn),
               "single_cos_target_mean": sum(r["single"]["cos_target"] for r in rayn) / len(rayn),
               "rayn_max_norm_over_single": sum(r["rayn"]["max_write_norm"] / r["single"]["max_write_norm"]
                                                for r in rayn) / len(rayn)}
    print("SUMMARY", json.dumps(metrics))
    log_run("e01_persistence", vars(a), metrics, out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl",
            smoke=a.smoke)


if __name__ == "__main__":
    main()
