"""E15 - Tier M: certified test-time memory read INSIDE a frozen pretrained LM (mid-layer residual stream).

Substrate (src/tabayyun/models/midlayer.py).  Memory key: whitened hidden state at layer l = L/2 of the
last prompt token (the E10 key map).  Memory value: a residual-stream edit c = delta*/R that, added after
block l, makes the frozen LM emit the asserted object.  Serving: the read y is snapped to the nearest
value of the codebook C (all values asserted in the trial, the attacker's included); a certified snap
is injected as R * c at layer l and the remaining L - l blocks produce the next token.

Facts, sources, cells, attacks and defences are those of E10 (CounterFact records 2000-3999; the first
f sources assert another record's object on two target facts).  The defences run through
`e04.apply_defences` with predict(y) = <y, C> - |C|^2/2 and W = C, so read-level certificates and
violations are computed exactly as on the other tiers.  The quantities that matter here are measured on
the LM's OUTPUT after the remaining blocks:

  final_recall            next token = asserted object (benign facts)
  final_installed         next token = attacker's object (targets)
  final_certified_correct certified snap and next token = object (all facts, clean)
  final_violations        certified snap but next token != the next token under the attack-free reference
  raw_transfer_failures   (tabayyun_w, no snapping) read-level certificate holds but the unsnapped read
                          gives a different next token than the unsnapped reference read

Usage:
    python experiments/e15_midlayer.py --model gpt2 --smoke
    python experiments/e15_midlayer.py --model gpt2-xl --device cuda --tune_w
    python experiments/e15_midlayer.py --model gpt2-xl --device cuda --trials 30 --trial_offset 1500 --tag gpt2xl_confirm
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
import e04_mqar_attack as e04  # noqa: E402
import e10_lm_retrofit as e10  # noqa: E402
from tabayyun import Gates  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Unbounded  # noqa: E402
from tabayyun.certify import action_certified  # noqa: E402
from tabayyun.memory import LinearMemory  # noqa: E402
from tabayyun.models.midlayer import MidLayerLM  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

DEFS = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed", "robustrag", "smsr", "tma_corr",
        "tabayyun", "tabayyun_strict", "tabayyun_w"]
ATTACKS = ["none", "assert", "edge", "unbounded"]
STEPS, LR = 30, 0.1                     # value search, fixed before any run


class Tier:
    def __init__(self, model, device):
        L = __import__("transformers").AutoConfig.from_pretrained(model).num_hidden_layers
        self.layer = L // 2
        self.S = e10.Substrate(model, device, self.layer, 50.0)        # keys at layer L/2 (E10 key map)
        self.lm = MidLayerLM(model, device)
        self.R = float(self.S.hid[self.layer][:e10.N_CALIB].norm(dim=-1).median())
        self.prompts = [r["prompt"] for r in self.S.recs]
        self.cache_file = ROOT / "checkpoints" / f"e15_values_{model.replace('/', '_')}_L{self.layer}.pt"
        self.V = torch.load(self.cache_file) if self.cache_file.exists() else {}
        self.device = device

    def values(self, pairs):
        """Unit-scale values c = delta*/R for (record, token) pairs, cached on disk."""
        miss = [p for p in dict.fromkeys(pairs) if p not in self.V]
        if miss:
            D = self.lm.value_vectors([self.prompts[i] for i, _ in miss], [t for _, t in miss], self.layer, self.R,
                                      steps=STEPS, lr=LR)
            for p, d in zip(miss, D):
                self.V[p] = d / self.R
        return torch.stack([self.V[p] for p in pairs]).to(self.device)

    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.V, self.cache_file)

    def precompute(self, batch=256):
        pool = self.S.pool.tolist()
        for i in range(0, len(pool), batch):
            self.values([(r, int(self.S.tok_new[r])) for r in pool[i:i + batch]])
            print(f"  values {min(i + batch, len(pool))}/{len(pool)}", flush=True)
        self.save()


def run_trial(T: Tier, F, n, f, rho, regime, attack, seed, defences, rounds):
    S = T.S
    facts, tok, adv_tok, targets, honest_idx, g = e10.build_trial(S, F, n, f, rho, regime, seed, rounds)
    facts_l = facts.tolist()
    pairs = [(facts_l[j], int(tok[j])) for j in range(F)] + [(facts_l[t], int(adv_tok[t])) for t in targets]
    C = T.values(pairs)                                                  # (m, d) codebook
    idx = {(j, int(tok[j])): j for j in range(F)}
    idx.update({(t, int(adv_tok[t])): F + i for i, t in enumerate(targets)})
    vals = torch.arange(F)
    adv_vals = vals.clone()
    for i, t in enumerate(targets):
        adv_vals[t] = F + i
    asserts = e10.assertions(F, n, f, tok, adv_tok, targets, honest_idx, attack, g)
    Kf = S.K[facts.to(S.device)]
    dv = C.shape[1]
    Q = LinearMemory(n, S.dk, dv, e10.Q_GATES, device=S.device)
    written = []
    for t in range(max((len(a) for a in asserts), default=0)):
        k = torch.zeros(n, S.dk, device=S.device); v = torch.zeros(n, dv, device=S.device)
        mask = torch.zeros(n, dtype=torch.bool, device=S.device)
        for s, a in enumerate(asserts):
            if t < len(a):
                j, tk = a[t]
                k[s], v[s], mask[s] = Kf[j], C[idx[(j, tk)]], True
                written.append((s, Kf[j]))
        Q.write(k, v, mask)

    def titans():
        flat = [(j, tk) for a in asserts for (j, tk) in a]
        M = LinearMemory(1, S.dk, dv, e10.T_GATES, device=S.device)
        for i in torch.randperm(len(flat), generator=g).tolist():
            j, tk = flat[i]
            M.write(Kf[j][None], C[idx[(j, tk)]][None])
        return M.M[0]

    hook = None
    if attack in ("edge", "unbounded"):
        ks, vs = Kf[targets[0]], C[int(adv_vals[targets[0]])]
        hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                else Unbounded(list(range(f)), ks, vs, scale=20.0))
    benign = [j for j in range(F) if j not in targets and any(j in honest_idx[s] for s in honest_idx)]
    bias = 0.5 * (C * C).sum(1)
    res = e04.apply_defences(defences, Q.M, written, Kf, titans, lambda y: y @ C.T - bias, lambda: C, hook, n, f, F,
                             seed, S.device, vals, adv_vals, targets, benign, attack, certify_fn=action_certified,
                             keep=True)
    # ---- downstream: inject at layer l and run the remaining blocks (one batched pass)
    prompts = [T.prompts[i] for i in facts_l]
    jobs = [("oracle", C[vals]), ("none", torch.zeros(F, dv, device=S.device))]
    for d, r in res.items():
        jobs.append((d, C[r["_pred"].to(S.device)]))
        if "_cert" in r:
            jobs.append((d + "|ref", C[r["_ref"].to(S.device)]))
        if d == "tabayyun_w":
            jobs += [(d + "|raw", r["_y"]), (d + "|rawref", r["_yref"])]
    D = torch.cat([x for _, x in jobs]) * T.R
    nxt = T.lm.next_token(prompts * len(jobs), T.layer, D).view(len(jobs), F)
    out = {name: nxt[i] for i, (name, _) in enumerate(jobs)}
    tok_c, adv_c = tok.cpu(), adv_tok.cpu()

    def score(o):
        e = {"final_recall": float((o[benign] == tok_c[benign]).float().mean()) if benign else float("nan"),
             "final_installed": float(np.mean([int(o[t] == adv_c[t]) for t in targets])) if attack != "none" else None,
             "final_amnesia": float(np.mean([int(o[t] != tok_c[t]) for t in targets]))}
        return e

    rows = {}
    for d, r in res.items():
        e = {k: v for k, v in r.items() if not k.startswith("_")}
        e.update(score(out[d]))
        if "_cert" in r:
            c = r["_cert"]
            e["final_certified_correct"] = float((c & (out[d] == tok_c)).float().mean())
            e["final_served_installed"] = (float(np.mean([int(bool(c[t]) and out[d][t] == adv_c[t]) for t in targets]))
                                           if attack != "none" else None)
            e["final_violations"] = int((c & (out[d] != out[d + "|ref"])).sum())
            if d == "tabayyun_w":
                e["raw_transfer_failures"] = int((c & (out[d + "|raw"] != out[d + "|rawref"])).sum())
                e["n_certified"] = int(c.sum())
                rows["tabayyun_w_raw"] = score(out[d + "|raw"])
        rows[d] = e
    rows["oracle_value"] = score(out["oracle"])
    rows["no_memory"] = score(out["none"])
    return rows


def tune_w(T, a):
    """Clean-only eps for Tabayyun-W (attack = none, exploratory trials 0-4); objective: final clean recall +
    final certified-and-correct; the smallest eps within 0.02 of the best (the rule of every other tier)."""
    out, grid = {}, (0.1, 0.2, 0.3, 0.4, 0.6)
    for eps in grid:
        e04.EPS_W = eps
        rec, cc = [], []
        for (rg, n, f, rho, R) in e10.cells_for([1, 3]):
            for s in range(5):
                r = run_trial(T, a.facts, n, f, rho, rg, "none", s, ["tabayyun_w"], R)["tabayyun_w"]
                rec.append(r["final_recall"]); cc.append(r["final_certified_correct"])
        out[str(eps)] = {"final_recall": float(np.nanmean(rec)), "final_certified_correct": float(np.mean(cc)),
                         "objective": float(np.nanmean(rec) + np.mean(cc))}
        print(f"eps={eps} {out[str(eps)]}", flush=True)
    best = max(v["objective"] for v in out.values())
    out["selected_eps"] = min(e for e in grid if out[str(e)]["objective"] >= best - 0.02)
    return out


def summarise(rows, cells, atks):
    summ = {}
    ax = [x for x in atks if x != "none"]
    names = sorted({r["defence"] for r in rows})
    for (rg, n, f, rho, R), d in itertools.product(cells, names):
        rr = [r for r in rows if (r["regime"], r["f"], r["rho"], r["R"], r["defence"]) == (rg, f, rho, R, d)]
        if not rr:
            continue
        cl = [r for r in rr if r["attack"] == "none"]
        at = [r for r in rr if r["attack"] != "none"]
        m = lambda xs, k: float(np.nanmean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
        e = {"final_recall": m(cl, "final_recall"),
             "worst_final_installed": max((m([r for r in at if r["attack"] == x], "final_installed") for x in ax),
                                          default=float("nan")),
             "worst_final_amnesia": max((m([r for r in at if r["attack"] == x], "final_amnesia") for x in ax),
                                        default=float("nan"))}
        if "final_violations" in rr[0]:
            e.update({"final_certified_correct_clean": m(cl, "final_certified_correct"),
                      "worst_final_served_installed": max((m([r for r in at if r["attack"] == x], "final_served_installed")
                                                           for x in ax), default=float("nan")),
                      "final_violations": int(sum(r["final_violations"] for r in rr)),
                      "read_violations": int(sum(r["violations"] + r["action_violations"] for r in rr))})
            if "raw_transfer_failures" in rr[0]:
                e["raw_transfer_failures"] = int(sum(r["raw_transfer_failures"] for r in rr))
                e["n_certified"] = int(sum(r["n_certified"] for r in rr))
        summ[f"{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune_w", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--trial_offset", type=int, default=1500)
    ap.add_argument("--facts", type=int, default=16)
    ap.add_argument("--obs_rounds", default="1,3")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = ROOT / "results" / "e15m"; out.mkdir(parents=True, exist_ok=True)
    T = Tier(a.model, a.device)
    safe = a.model.replace("/", "_")
    print(f"model={a.model} layer={T.layer}/{T.lm.n_layers} R={T.R:.2f} dk={T.S.dk} pool={len(T.S.pool)}", flush=True)
    w_file = out / f"tuning_w_{safe}.json"
    if not a.smoke:
        T.precompute()
    if a.tune_w:
        res = tune_w(T, a)
        w_file.write_text(json.dumps(res, indent=1))
        T.save()
        print("selected eps", res["selected_eps"])
        return
    if w_file.exists():
        e04.EPS_W = json.loads(w_file.read_text())["selected_eps"]
    if a.smoke:
        cells, atks, trials = e10.cells_for(None, True), ["none", "assert", "edge"], range(2)
    else:
        cells = e10.cells_for([int(x) for x in a.obs_rounds.split(",")])
        atks, trials = ATTACKS, range(a.trial_offset, a.trial_offset + a.trials)
    rows, t0 = [], time.time()
    for (rg, n, f, rho, R), at in itertools.product(cells, atks):
        for s in trials:
            for d, r in run_trial(T, a.facts, n, f, rho, rg, at, s, DEFS, R).items():
                rows.append({"regime": rg, "n": n, "f": f, "rho": rho, "R": R, "attack": at, "trial": s,
                             "defence": d, **r})
        print(f"{rg} n={n} f={f} rho={rho} R={R} {at}: {time.time() - t0:.0f}s", flush=True)
    T.save()
    name = "smoke" if a.smoke else f"e15m{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = summarise(rows, cells, atks)
    for k, v in summ.items():
        print(f"{k:42} " + " ".join(f"{m}={x:.2f}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    fv = sum(v.get("final_violations", 0) for v in summ.values())
    rv = sum(v.get("read_violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "final_violations": fv, "read_violations": rv,
                                 "seconds": round(time.time() - t0, 1)}))
    log_run("e15_midlayer", {**vars(a), "layer": T.layer, "R": T.R, "eps_w": e04.EPS_W},
            {"summary": summ, "final_violations": fv, "read_violations": rv},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
