"""E02 - Hypotheses 2-3 on the controlled multi-source testbed.

Compares Tabayyun-Cert with every single-client baseline under every attack,
and records certificate soundness / tightness (true drift vs certified drift).

Usage:
    python experiments/e02_synthetic_defence.py --smoke
    python experiments/e02_synthetic_defence.py --device cuda --seeds 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabayyun import GateConfig, Gates, TabayyunConfig, TabayyunMemory  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Rayn, TargetedWrite, Unbounded  # noqa: E402
from tabayyun.baselines import RobustAggMemory, SharedMemory  # noqa: E402
from tabayyun.data import FactWorld, WorldConfig, run_round  # noqa: E402
from tabayyun.metrics import benign_recall, target_cos, targeted_success  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DEFENCES = ["titans", "titans_clip", "titans_ttl", "agg_mean", "agg_trmean", "agg_median",
            "agg_krum", "agg_geomed", "agg_cclip", "tabayyun"]
ATTACKS = ["none", "targeted", "single_shot", "rayn", "edge_of_range", "unbounded"]


def build(defence, wc: WorldConfig, f, k, device):
    gates = Gates(theta=0.5, eta=0.0, alpha=0.02)
    n, dk, dv, tau = wc.n_sources, wc.dk, wc.dv, wc.tau
    if defence == "titans":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=0.02), device=device)
    if defence == "titans_clip":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.5, alpha=0.02, clip=0.5), device=device)
    if defence == "titans_ttl":
        return SharedMemory(dk, dv, Gates(theta=0.5, eta=0.0, alpha=0.2), device=device)
    if defence.startswith("agg_"):
        return RobustAggMemory(n, dk, dv, gates, agg=defence[4:], f=f, tau=tau, device=device)
    if defence == "tabayyun":
        return TabayyunMemory(TabayyunConfig(n_sources=n, dk=dk, dv=dv, tau=tau, q_gates=gates,
                                             gate=GateConfig(k=k, f=f, gamma=0.3, nu=0.05),
                                             device=device))
    raise ValueError(defence)


def make_attack(name, w: FactWorld, mem, k_star, v_star):
    """Returns (write_attack, hook_attack)."""
    if name == "none":
        return None, None
    if name == "targeted":
        return TargetedWrite(w.adv, k_star, v_star), None
    if name == "single_shot":
        return TargetedWrite(w.adv, k_star, v_star, scale=10.0, steps=[0]), None
    if name == "rayn":
        return Rayn(w.adv, k_star, v_star, eps=0.1), None
    # promotion-level white-box attacks: only meaningful for quarantine defences;
    # for shared memory they degrade to the strongest write attack.
    if name in ("edge_of_range", "unbounded"):
        if isinstance(mem, TabayyunMemory):
            hook = (EdgeOfRange(w.adv, w.honest, k_star, v_star, nu=0.05) if name == "edge_of_range"
                    else Unbounded(w.adv, k_star, v_star))
            return None, hook
        return TargetedWrite(w.adv, k_star, v_star, scale=10.0 if name == "unbounded" else 1.0), None
    raise ValueError(name)


def run_one(defence, attack, wc: WorldConfig, f, k, rounds, device):
    w = FactWorld(wc, device=device)
    mem = build(defence, wc, f, k, device)
    target_fact = 0
    k_star, v_star, v_true = w.keys[target_fact], -w.values[target_fact], w.values[target_fact]
    watk, hook = make_attack(attack, w, mem, k_star, v_star)
    if isinstance(mem, TabayyunMemory):
        mem.delta_hook = hook
        mem.oracle_honest = w.honest
    sound = True
    for _ in range(rounds):
        run_round(mem, w, watk)
        if isinstance(mem, TabayyunMemory):
            sound &= mem.true_drift() <= mem.certificate * (1 + 1e-4) + 1e-5
    benign = torch.arange(1, wc.n_facts)
    shared = benign[~w.unique[1:]]
    res = {
        "asr": float(targeted_success(mem, k_star, v_star, v_true)) if attack != "none" else float("nan"),
        "target_cos": target_cos(mem, k_star, v_star),
        "recall_high_risk_shared": benign_recall(mem, w.keys[shared], w.values[shared], extra_candidates=v_star[None]),
        "recall_high_risk_all": benign_recall(mem, w.keys[benign], w.values[benign], extra_candidates=v_star[None]),
    }
    if isinstance(mem, TabayyunMemory):
        q = w.keys
        res.update({"cert_sound": sound, "cert_frob": mem.certificate, "true_drift": mem.true_drift(),
                    "tightness_ratio": mem.certificate / max(mem.true_drift(), 1e-9),
                    "promoted_frac": float(mem.last.promoted.float().mean()),
                    "cert_query_mean": float(mem.query_certificate(q).mean())})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--out", default="results/e02")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    defs = DEFENCES if not a.smoke else ["titans", "agg_mean", "agg_trmean", "tabayyun"]
    atks = ATTACKS if not a.smoke else ["none", "targeted", "edge_of_range"]
    seeds = range(1 if a.smoke else a.seeds)
    fracs = [0.2] if a.smoke else [0.1, 0.2, 0.3]
    rows = []
    n = 10
    for d, at, fr, s in itertools.product(defs, atks, fracs, seeds):
        f = max(1, int(round(fr * n)))
        k = 2 * f + 1
        wc = WorldConfig(n_sources=n, n_adv=f, dk=32, dv=32, n_facts=24, rho=0.7, sigma=0.1, tau=50, seed=s)
        r = run_one(d, at, wc, f, k, 3 if a.smoke else a.rounds, a.device)
        r.update({"defence": d, "attack": at, "f_over_n": fr, "seed": s})
        rows.append(r)
        print(json.dumps(r))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / ("smoke.json" if a.smoke else "e02.json")).write_text(json.dumps(rows, indent=1))
    tab = [r for r in rows if r["defence"] == "tabayyun"]
    metrics = {"tabayyun_cert_sound_all": all(r["cert_sound"] for r in tab),
               "n_runs": len(rows)}
    for d in defs:
        rr = [r for r in rows if r["defence"] == d and r["attack"] != "none"]
        if rr:
            metrics[f"asr_{d}"] = sum(r["asr"] for r in rr) / len(rr)
        cl = [r for r in rows if r["defence"] == d and r["attack"] == "none"]
        if cl:
            metrics[f"clean_recall_{d}"] = sum(r["recall_high_risk_shared"] for r in cl) / len(cl)
    print("SUMMARY", json.dumps(metrics))
    log_run("e02_synthetic_defence", vars(a), metrics,
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
