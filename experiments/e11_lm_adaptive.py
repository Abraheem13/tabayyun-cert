"""E11 - adaptive gradient attacker (PGD / BPDA) against the Tier-D memory on frozen pretrained LMs.

Same attacker and code as E05 (`e05.attack_all`): the adversary knows the LM, the gate and
every honest quarantine, and sets the states of its own f sources arbitrarily to make the LM
emit its object at the two target prompts.  Substrate, facts and trials as in E10
(CounterFact, logits = W_U (x_out + gain * M k)), with the clean-tuned (layer, boost) of E10.

Usage:
    python experiments/e11_lm_adaptive.py --model gpt2 --smoke
    python experiments/e11_lm_adaptive.py --model gpt2-xl --trials 30 --trial_offset 900 --tag gpt2xl_confirm
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
import e05_adaptive as e05  # noqa: E402
import e10_lm_retrofit as e10  # noqa: E402
from tabayyun.certify import action_certified_gram  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402


class Shim:
    """What the attacker needs from the substrate: logits given a read, and the readout matrix."""

    def __init__(self, S):
        self.S = S

    def logits_from_read(self, h, y):
        return (h + self.S.gain * y) @ self.S.W_U.T

    def readout_matrix(self):
        return self.S.gain * self.S.W_U


UNION_DEFS = ["tabayyun", "tabayyun_strict", "agg_trmean"]


def run(S, F, n, f, rho, regime, R, seed, steps, lr, union=False):
    facts, vals, adv_vals, targets, hidx, g = e10.build_trial(S, F, n, f, rho, regime, seed, R)
    Kf = S.K[facts.to(S.device)]
    x_q = S.x_out[facts.to(S.device)]
    asserts = e10.assertions(F, n, f, vals, adv_vals, targets, hidx, "none", g)   # adversary writes nothing:
    with torch.no_grad():                                                          # it sets its rows directly
        Ms, written, _ = e10.write_memories(S, Kf, asserts, n, g)
    ks, vs = Kf[targets[0]], S.E_hat[int(adv_vals[targets[0]])]
    adv_written = [(s, Kf[t]) for s in range(f) for t in targets]
    adv_states = None
    if union:      # Amendment 12: the adversary's REAL quarantines under the static write attacks, as extra starts
        adv_states = []
        with torch.no_grad():
            for i, at in enumerate(("assert", "assert_x4")):
                g2 = torch.Generator().manual_seed(seed * 7919 + i + 1)
                a2 = e10.assertions(F, n, f, vals, adv_vals, targets, hidx, at, g2)
                Madv, _, _ = e10.write_memories(S, Kf, a2[:f] + [[] for _ in range(n - f)], n, g2)
                adv_states.append(Madv[:f])
                # the adversary also writes camouflage keys; they count as writers for anchoring, as in E10
                adv_written = adv_written + [(s_, Kf[j]) for s_ in range(f) for (j, _tk) in a2[s_]]
    torch.manual_seed(seed)                                                        # random restart init
    return e05.attack_all(Shim(S), Ms, written, adv_written, Kf, x_q, vals, adv_vals, targets, ks, vs,
                          f, n, steps, lr, certify_fn=action_certified_gram,
                          defs=UNION_DEFS if union else None, adv_states=adv_states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--trial_offset", type=int, default=900)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--defences", default=None, help="comma list for --union (default: UNION_DEFS)")
    ap.add_argument("--union", action="store_true",
                    help="Amendment 12 union attacker: static write attacks + PGD with/without the cone hinge")
    a = ap.parse_args()
    out = ROOT / "results" / "e11"; out.mkdir(parents=True, exist_ok=True)
    tune_file = ROOT / "results" / "e10" / f"tuning_{a.model.replace('/', '_')}.json"
    if not tune_file.exists():
        sys.exit(f"no clean-tuned (layer, boost) for {a.model}: run e10_lm_retrofit.py --tune first")
    sel = json.loads(tune_file.read_text())["selected"]
    w_file = ROOT / "results" / "e10" / f"tuning_w_{a.model.replace('/', '_')}.json"
    if w_file.exists():                        # clean-tuned eps of tabayyun_w (shared e04 module)
        e10.e04.EPS_W = json.loads(w_file.read_text())["selected_eps"]
    if a.defences:
        UNION_DEFS[:] = a.defences.split(",")
    S = e10.Substrate(a.model, a.device, sel["layer"], sel["boost"])
    print(f"model={a.model} key_layer={S.layer} boost={S.boost} dk={S.dk}", flush=True)
    cells = [(rg, n, f, rho, R) for rg, n, fs in [("dense", 10, (1, 2)), ("locality", 20, (2, 3))]
             for f in fs for rho in (0.4, 0.7) for R in (1, 3)]
    trials = range(a.trial_offset, a.trial_offset + a.trials)
    if a.smoke:
        cells, trials, a.steps = cells[:2] + cells[-2:], range(2), 30
        if a.union:
            cells, trials = [cells[0], ("locality", 20, 3, 0.4, 1)], range(4)
    rows, t0 = [], time.time()
    for i, ((rg, n, f, rho, R), s) in enumerate(itertools.product(cells, trials)):
        res = run(S, 16, n, f, rho, rg, R, s, a.steps, a.lr, a.union)
        rows.extend({"regime": rg, "f": f, "rho": rho, "R": R, "trial": s, "defence": d, **r} for d, r in res.items())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cells) * len(trials)} trials  {time.time() - t0:.0f}s", flush=True)
    name = ("smoke_union" if a.union else "smoke") if a.smoke else f"e11{('_' + a.tag) if a.tag else ''}"
    defs = UNION_DEFS if a.union else e05.DEFS
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    for (rg, n, f, rho, R) in cells:
        for d in defs:
            rr = [r for r in rows if (r["regime"], r["f"], r["rho"], r["R"], r["defence"]) == (rg, f, rho, R, d)]
            e = {"pgd_installed": float(np.mean([r["installed"] for r in rr])),
                 "edge_installed": float(np.mean([r["edge_installed"] for r in rr]))}
            if a.union:
                e["static_installed"] = float(np.mean([r["static_installed"] for r in rr]))
                e["amnesia_mean"] = float(np.mean([r["amnesia"] for r in rr]))
                if any("served_installed" in r for r in rr):
                    e["served_installed"] = float(np.mean([r["served_installed"] for r in rr]))
                    e["served_correct"] = float(np.mean([r["served_correct"] for r in rr]))
            if any("violations" in r for r in rr):
                e["violations"] = int(sum(r.get("violations", 0) + r.get("action_violations", 0) for r in rr))
            summ[f"{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
            print(f"{rg:8} f={f} rho={rho} R={R} {d:15} " + " ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = sum(v.get("violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "seconds": round(time.time() - t0, 1)}))
    log_run("e11_lm_adaptive", vars(a), {"summary": summ, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
