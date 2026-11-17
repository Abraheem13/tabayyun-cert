"""E13 - capacity and cost scaling of the Tier-D memory on a frozen pretrained LM (CounterFact).

How do security, recall, certification and wall-clock cost change when one memory holds
F = 16 ... 1024 facts?  Substrate, facts, gate and defence code are those of E10
(`e10.build_trial`, `e10.write_memories`, `e04.apply_defences`), with a larger key space
(d_k = 1024 whitened dimensions, fitted on the 2,000 calibration prompts).

Per (F, cell, attack, trial) and defence: recall, installed, amnesia, certified-and-correct,
violations, served-installed, and the wall-clock time of the defence's combine + certificate
step (the quarantine writes are timed separately, once per trial).

Usage:
    python experiments/e13_scale.py --model gpt2 --smoke
    python experiments/e13_scale.py --model gpt2-xl --trials 20 --trial_offset 1300 --tag gpt2xl_scale
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import e10_lm_retrofit as e10  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Unbounded  # noqa: E402
from tabayyun.certify import action_certified_gram  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

e04 = e10.e04
DEFS = ["titans", "agg_trmean", "agg_median", "agg_geomed", "robustrag", "tabayyun_strict", "tabayyun_w"]
CELLS = [("dense", 10, 2, 0.7, 3), ("locality", 20, 3, 0.7, 3), ("locality", 20, 3, 0.4, 1)]
ATTACKS = ["none", "assert", "edge", "unbounded"]
FACTS = [16, 64, 256, 1024]
TUNE_GRID = (0.2, 0.3, 0.4, 0.6, 0.8, 1.0)


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def run_trial(S, F, n, f, rho, regime, attack, seed, R):
    facts, vals, adv_vals, targets, honest_idx, g = e10.build_trial(S, F, n, f, rho, regime, seed, R)
    Kf = S.K[facts.to(S.device)]
    x_q = S.x_out[facts.to(S.device)]
    asserts = e10.assertions(F, n, f, vals, adv_vals, targets, honest_idx, attack, g)
    _sync(S.device); t0 = time.time()
    Ms, written, titans = e10.write_memories(S, Kf, asserts, n, g)
    _sync(S.device); t_write = time.time() - t0
    # anchoring evidence = distinct (source, key) pairs; identical keys repeat in Tier D, so this
    # keeps the anchored basis the same while keeping the key-history matrix small at large F
    wdedup = [(s_, Kf[j]) for s_, lst in enumerate(asserts) for j in sorted({j for j, _ in lst})]
    hook = None
    if attack in ("edge", "unbounded"):
        ks, vs = Kf[targets[0]], S.E_hat[int(adv_vals[targets[0]])]
        hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                else Unbounded(list(range(f)), ks, vs, scale=20.0))
    benign = [j for j in range(F) if j not in targets and any(j in honest_idx[s] for s in honest_idx)]
    out = {}
    for d in DEFS:
        _sync(S.device); t0 = time.time()
        r = e04.apply_defences([d], Ms, wdedup, Kf, titans, lambda y: (x_q + S.gain * y) @ S.W_U.T,
                               lambda: S.gain * S.W_U, hook, n, f, F, seed, S.device, vals, adv_vals, targets,
                               benign, attack, certify_fn=action_certified_gram)[d]
        _sync(S.device)
        r["seconds"] = time.time() - t0
        r["write_seconds"] = t_write
        out[d] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial_offset", type=int, default=1300)
    ap.add_argument("--dk", type=int, default=1024)
    ap.add_argument("--facts", default=",".join(map(str, FACTS)))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune_w", action="store_true",
                    help="clean-only eps of tabayyun_w per memory size F (trials 0-4, attack = none)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    safe = a.model.replace("/", "_")
    tf = ROOT / "results" / "e10" / f"tuning_{safe}.json"
    wf = ROOT / "results" / "e10" / f"tuning_w_{safe}.json"
    if not tf.exists() or not wf.exists():
        sys.exit(f"no clean-tuned (layer, boost) and eps for {a.model}: run e10 --tune and --tune_w first")
    sel = json.loads(tf.read_text())["selected"]
    e04.EPS_W = json.loads(wf.read_text())["selected_eps"]
    e10.DK = a.dk
    S = e10.Substrate(a.model, a.device, sel["layer"], sel["boost"])
    facts = [int(x) for x in a.facts.split(",")]
    out = ROOT / "results" / "e13"; out.mkdir(parents=True, exist_ok=True)
    tw = out / f"tuning_w_{safe}.json"
    if a.tune_w:
        global DEFS
        DEFS, res = ["tabayyun_w"], {}
        for F in facts:
            grid = {}
            for eps in TUNE_GRID:
                e04.EPS_W = eps
                rec, cc = [], []
                for (rg, n, f, rho, R) in CELLS[:2]:
                    for s in range(5):
                        r = run_trial(S, F, n, f, rho, rg, "none", s, R)["tabayyun_w"]
                        rec.append(r["recall"]); cc.append(r["certified_correct_frac"])
                grid[str(eps)] = float(np.nanmean(rec) + np.mean(cc))
                print(f"F={F} eps={eps} recall={np.nanmean(rec):.3f} cert_correct={np.mean(cc):.3f}", flush=True)
            best = max(grid.values())
            res[str(F)] = {"objective": grid, "selected_eps": min(e for e in TUNE_GRID if grid[str(e)] >= best - 0.02)}
            print(f"F={F} selected eps {res[str(F)]['selected_eps']}", flush=True)
        tw.write_text(json.dumps(res, indent=1))
        return
    eps_by_F = json.loads(tw.read_text()) if tw.exists() else {}
    cells, atks, trials = CELLS, ATTACKS, range(a.trial_offset, a.trial_offset + a.trials)
    if a.smoke:
        facts, cells, atks, trials = [x for x in facts if x <= 256], CELLS[:2], ["none", "edge"], range(2)
    print(f"model={a.model} dk={S.dk} layer={S.layer} boost={S.boost} eps_w_by_F="
          f"{ {F: eps_by_F.get(str(F), {}).get('selected_eps') for F in facts} } facts={facts}", flush=True)
    rows, t0 = [], time.time()
    for F, (rg, n, f, rho, R), at in itertools.product(facts, cells, atks):
        e04.EPS_W = eps_by_F.get(str(F), {}).get("selected_eps", e04.EPS_W)
        for s in trials:
            for d, r in run_trial(S, F, n, f, rho, rg, at, s, R).items():
                rows.append({"F": F, "regime": rg, "n": n, "f": f, "rho": rho, "R": R, "attack": at, "trial": s,
                             "defence": d, **r})
        print(f"F={F} {rg} f={f} rho={rho} R={R} {at}: {time.time() - t0:.0f}s", flush=True)
    name = "smoke" if a.smoke else f"e13{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    for F, (rg, n, f, rho, R), d in itertools.product(facts, cells, DEFS):
        rr = [r for r in rows if (r["F"], r["regime"], r["f"], r["rho"], r["R"], r["defence"]) == (F, rg, f, rho, R, d)]
        cl = [r for r in rr if r["attack"] == "none"]
        att = [r for r in rr if r["attack"] != "none"]
        e = {"clean_recall": float(np.nanmean([r["recall"] for r in cl])),
             "worst_installed": float(max(np.mean([r["installed"] for r in att if r["attack"] == x]) for x in atks[1:])),
             "seconds_combine": float(np.mean([r["seconds"] for r in rr])),
             "seconds_write": float(np.mean([r["write_seconds"] for r in rr]))}
        if d in e04.CERT_METHODS:
            e.update({"violations": int(sum(r["violations"] + r["action_violations"] for r in rr)),
                      "certified_correct_clean": float(np.mean([r["certified_correct_frac"] for r in cl])),
                      "worst_served_installed": float(max(np.mean([r["served_installed"] for r in att if r["attack"] == x])
                                                          for x in atks[1:]))})
        if d == "tabayyun_w":
            e["eps_w"] = eps_by_F.get(str(F), {}).get("selected_eps")
        summ[f"F={F}|{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
        print(f"F={F:5d} {rg:8} f={f} rho={rho} R={R} {d:15} " + " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()), flush=True)
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = sum(v.get("violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "seconds": round(time.time() - t0, 1)}))
    log_run("e13_scale", vars(a), {"summary": summ, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- sharded memory (Amendment 18)
def shard_of(K: torch.Tensor, S: int, seed: int = 12345) -> torch.Tensor:
    """Deterministic routing of keys to S shards: the shard is the argmax of S fixed random
    projections of the (unit) key.  Writers and readers route a key the same way, so every
    shard is an independent Tabayyun instance with ~F/S facts (Theorem 6 and Prop. R apply
    per shard, and the per-query certificate is that of the query's shard)."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(K.shape[1], S, generator=g).to(K)
    return (K @ P).argmax(1)


def run_trial_sharded(S, F, n, f, rho, regime, attack, seed, R, shards):
    facts, vals, adv_vals, targets, honest_idx, g = e10.build_trial(S, F, n, f, rho, regime, seed, R)
    Kf = S.K[facts.to(S.device)]
    x_q = S.x_out[facts.to(S.device)]
    asserts = e10.assertions(F, n, f, vals, adv_vals, targets, honest_idx, attack, g)
    sh = shard_of(Kf, shards).cpu()
    acc = {d: {"rec_n": 0, "rec_s": 0.0, "inst": [], "served": [], "cc_s": 0.0, "viol": 0, "sec": 0.0} for d in DEFS}
    t_write = 0.0
    benign_all = [j for j in range(F) if j not in targets and any(j in honest_idx[s] for s in honest_idx)]
    for b in range(shards):
        J = [j for j in range(F) if int(sh[j]) == b]
        if not J:
            continue
        pos = {j: i for i, j in enumerate(J)}
        sub_asserts = [[(pos[j], tk) for (j, tk) in lst if j in pos] for lst in asserts]
        tg = [pos[t] for t in targets if t in pos]
        sub_hidx = {s: [pos[j] for j in honest_idx[s] if j in pos] for s in honest_idx}
        Kb, xb = Kf[J], x_q[J]
        _sync(S.device); t0 = time.time()
        Ms, _w, titans = e10.write_memories(S, Kb, sub_asserts, n, g)
        _sync(S.device); t_write += time.time() - t0
        wd = [(s_, Kb[j]) for s_, lst in enumerate(sub_asserts) for j in sorted({j for j, _ in lst})]
        hook = None
        if attack in ("edge", "unbounded") and tg:
            ks, vs = Kb[tg[0]], S.E_hat[int(adv_vals[targets[[pos.get(t, -1) for t in targets].index(tg[0])]])]
            hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                    else Unbounded(list(range(f)), ks, vs, scale=20.0))
        ben = [pos[j] for j in benign_all if j in pos]
        vb, ab = vals[J], adv_vals[J]
        for d in DEFS:
            _sync(S.device); t0 = time.time()
            r = e04.apply_defences([d], Ms, wd, Kb, titans, lambda y: (xb + S.gain * y) @ S.W_U.T,
                                   lambda: S.gain * S.W_U, hook, n, f, len(J), seed, S.device, vb, ab,
                                   tg if tg else [0], ben, attack if tg else "none",
                                   certify_fn=action_certified_gram)[d]
            _sync(S.device)
            A = acc[d]
            A["sec"] += time.time() - t0
            if ben and r["recall"] == r["recall"]:
                A["rec_s"] += r["recall"] * len(ben); A["rec_n"] += len(ben)
            if tg and attack != "none":
                A["inst"] += [r["installed"]] * len(tg)
                if "served_installed" in r:
                    A["served"] += [r["served_installed"]] * len(tg)
            if "violations" in r:
                A["viol"] += r["violations"] + r["action_violations"]
                A["cc_s"] += r["certified_correct_frac"] * len(J)
    out = {}
    for d, A in acc.items():
        r = {"recall": A["rec_s"] / max(A["rec_n"], 1), "seconds": A["sec"], "write_seconds": t_write,
             "installed": float(np.mean(A["inst"])) if A["inst"] else (None if attack == "none" else 0.0)}
        if d in e04.CERT_METHODS:
            r.update({"violations": A["viol"], "action_violations": 0, "certified_correct_frac": A["cc_s"] / F,
                      "served_installed": float(np.mean(A["served"])) if A["served"] else (None if attack == "none" else 0.0)})
        out[d] = r
    return out
