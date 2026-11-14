"""E10 - Tier D: poisoning a test-time memory retrofitted on a FROZEN pretrained LM (CounterFact).

Substrate (src/tabayyun/models/lm_retrofit.py):  logits = W_U (x_out + gain * M k).
Facts are CounterFact records; the knowledge to be stored is the record's NEW object
(prompt -> target_new), which the frozen LM does not know (base accuracy < 1%), so
recall measures the memory, not the LM's prior.  n sources assert facts at test time
(one Titans write per assertion); the first f are adversarial and assert a WRONG object
for two target facts (plus camouflage).  Protocol, cells, attacks, defences, metrics and
row format are those of E04 (Tier B), and the defences run through the same code
(`e04.apply_defences`), so `scripts/analyze_e04.py` scores this experiment unchanged.

Fact pool: data/counterfact/counterfact_4k.json; records 0-1999 fit the key-whitening
map (calibration, clean), records 2000-3999 are the trial pool.

Usage:
    python experiments/e10_lm_retrofit.py --model gpt2 --smoke
    python experiments/e10_lm_retrofit.py --model gpt2-xl --tune            # clean-only (layer, boost)
    python experiments/e10_lm_retrofit.py --model gpt2-xl --trials 100 --trial_offset 800 --obs_rounds 1,3 --tag xl_confirm
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun import Gates  # noqa: E402
from tabayyun.attacks import EdgeOfRange, Unbounded  # noqa: E402
from tabayyun.certify import action_certified_gram  # noqa: E402
from tabayyun.memory import LinearMemory  # noqa: E402
from tabayyun.models.lm_retrofit import KeyMap, LMRetrofit, cached_features  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("e04", ROOT / "experiments" / "e04_mqar_attack.py")
e04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e04)

DK = 256
N_CALIB = 2000
Q_GATES = Gates(theta=0.5, eta=0.0, alpha=0.005)       # per-source quarantines (as in Tier A)
T_GATES = Gates(theta=0.5, eta=0.5, alpha=0.005)       # ungated shared Titans memory (as in Tier A)


class Substrate:
    """Frozen-LM features of the fact pool + the key map; everything the trials need."""

    def __init__(self, model: str, device: str, layer: int | None, boost: float):
        recs = json.loads((ROOT / "data" / "counterfact" / "counterfact_4k.json").read_text(encoding="utf-8"))
        seen, keep = set(), []
        for i, r in enumerate(recs):                    # identical prompts would be identical keys
            if i < N_CALIB or r["prompt"] not in seen:
                keep.append(r)
            if i >= N_CALIB:
                seen.add(r["prompt"])
        self.recs = keep
        safe = model.replace("/", "_")
        probe = LMRetrofit.__new__(LMRetrofit)          # tokenizer only (cheap) for target tokens
        from transformers import AutoConfig, AutoTokenizer
        probe.tok = AutoTokenizer.from_pretrained(model)
        L = AutoConfig.from_pretrained(model).num_hidden_layers
        self.layer_grid = sorted({L // 3, L // 2, (2 * L) // 3, L})
        x_out, hid, W_U, _ = cached_features(model, [r["prompt"] for r in keep], self.layer_grid, device,
                                             ROOT / "checkpoints" / f"lmfeat_{safe}.pt")
        self.device, self.hid = device, hid
        self.x_out = x_out.to(device)
        self.W_U = W_U.to(device)
        E = self.W_U - self.W_U.mean(0, keepdim=True)
        self.E_hat = E / E.norm(dim=1, keepdim=True)
        # read gain in model-independent units: ``boost`` = logits added to the stored token
        self.unit = float(E.norm(dim=1).median())
        self.set_boost(boost)
        self.tok_new = torch.tensor([probe.first_token(r["target_new"]) for r in keep])
        self.tok_true = torch.tensor([probe.first_token(r["target_true"]) for r in keep])
        self.pool = torch.arange(N_CALIB, len(keep))
        self.set_layer(layer if layer is not None else L // 2)

    def set_boost(self, boost: float):
        self.boost, self.gain = boost, boost / self.unit

    def set_layer(self, layer: int):
        self.layer = layer
        km = KeyMap(self.hid[layer][:N_CALIB], DK)
        self.K = km(self.hid[layer]).to(self.device)                 # (N, dk) keys of every prompt
        self.dk = km.dk

    def base_accuracy(self):
        pred = (self.x_out[self.pool] @ self.W_U.T).argmax(-1).cpu()
        return {"target_new": float((pred == self.tok_new[self.pool]).float().mean()),
                "target_true": float((pred == self.tok_true[self.pool]).float().mean())}


def build_trial(S: Substrate, F, n, f, rho, regime, seed, rounds):
    g = torch.Generator().manual_seed(seed)
    facts = S.pool[torch.randperm(len(S.pool), generator=g)[:F]]      # pool indices of the F facts
    vals = S.tok_new[facts].clone()
    k_gate = 2 * f + 1
    honest = list(range(f, n))
    support = {}
    for j in range(F):
        if regime == "dense":
            support[j] = honest
        else:
            m = int(torch.randint(k_gate, (n - f) // 2 + 1, (1,), generator=g))
            support[j] = [honest[i] for i in torch.randperm(len(honest), generator=g)[:m].tolist()]
    targets = [0, 1]
    adv_vals = vals.clone()
    for t in targets:                                   # attacker's object: another record's object
        while True:
            c = int(S.tok_new[S.pool[int(torch.randint(len(S.pool), (1,), generator=g))]])
            if c != int(vals[t]):
                adv_vals[t] = c
                break
    honest_idx = {s: [] for s in honest}
    for _r in range(rounds):
        for j in range(F):
            for s in support[j]:
                if torch.rand(1, generator=g).item() < rho:
                    honest_idx[s].append(j)
    for t in targets:
        if not any(t in honest_idx[s] for s in honest):
            honest_idx[support[t][0]].append(t)
    return facts, vals, adv_vals, targets, honest_idx, g


def assertions(F, n, f, vals, adv_vals, targets, honest_idx, attack, g):
    """Per source: list of (fact j, token) in the order they are written."""
    out = []
    for _ in range(f):
        if attack == "none":
            out.append([])
            continue
        camo = [j for j in torch.randperm(F, generator=g)[: max(1, F // 3)].tolist() if j not in targets]
        reps = 4 if attack == "assert_x4" else 1
        lst = [(j, int(vals[j])) for j in camo] + [(t, int(adv_vals[t])) for t in targets] * reps
        out.append([lst[i] for i in torch.randperm(len(lst), generator=g).tolist()])
    for s in range(f, n):
        lst = [(j, int(vals[j])) for j in honest_idx[s]]
        out.append([lst[i] for i in torch.randperm(len(lst), generator=g).tolist()])
    return out


def write_memories(S: Substrate, Kf, asserts, n, g):
    """Per-source quarantines (n, dv, dk), the written keys, and a closure for the shared Titans memory."""
    dev, dv = S.device, S.W_U.shape[1]
    Q = LinearMemory(n, S.dk, dv, Q_GATES, device=dev)
    written = []
    for t in range(max((len(a) for a in asserts), default=0)):
        k = torch.zeros(n, S.dk, device=dev); v = torch.zeros(n, dv, device=dev)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        for s, a in enumerate(asserts):
            if t < len(a):
                j, tk = a[t]
                k[s], v[s], mask[s] = Kf[j], S.E_hat[tk], True
                written.append((s, Kf[j]))
        Q.write(k, v, mask)

    def titans():
        flat = [(j, tk) for a in asserts for (j, tk) in a]
        T = LinearMemory(1, S.dk, dv, T_GATES, device=dev)
        for i in torch.randperm(len(flat), generator=g).tolist():
            j, tk = flat[i]
            T.write(Kf[j][None], S.E_hat[tk][None])
        return T.M[0]

    return Q.M, written, titans


def run_trial(S: Substrate, F, n, f, rho, regime, attack, seed, defences, rounds):
    facts, vals, adv_vals, targets, honest_idx, g = build_trial(S, F, n, f, rho, regime, seed, rounds)
    Kf = S.K[facts.to(S.device)]                                       # (F, dk)
    x_q = S.x_out[facts.to(S.device)]                                  # (F, d)
    asserts = assertions(F, n, f, vals, adv_vals, targets, honest_idx, attack, g)
    Ms, written, titans = write_memories(S, Kf, asserts, n, g)
    hook = None
    if attack in ("edge", "unbounded"):
        ks, vs = Kf[targets[0]], S.E_hat[int(adv_vals[targets[0]])]
        hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                else Unbounded(list(range(f)), ks, vs, scale=20.0))
    benign = [j for j in range(F) if j not in targets and any(j in honest_idx[s] for s in honest_idx)]
    return e04.apply_defences(defences, Ms, written, Kf, titans, lambda y: (x_q + S.gain * y) @ S.W_U.T,
                              lambda: S.gain * S.W_U, hook, n, f, F, seed, S.device, vals, adv_vals, targets,
                              benign, attack, certify_fn=action_certified_gram)


def cells_for(obs_rounds, smoke=False):
    if smoke:
        return [("dense", 10, 2, 0.7, 1), ("locality", 20, 2, 0.7, 3)]
    return [(rg, n, f, rho, R) for rg, n, fs in [("dense", 10, (1, 2)), ("locality", 20, (2, 3))]
            for f in fs for rho in (0.4, 0.7) for R in obs_rounds]


def tune(S: Substrate, a):
    """Clean-only choice of (key layer, read boost): attack = none, exploratory trials 0-4.
    Objective fixed in advance: mean over the two Tabayyun variants and all cells of
    clean recall + certified-and-correct fraction."""
    out, best = {}, None
    for layer in S.layer_grid:
        S.set_layer(layer)
        for boost in (25.0, 50.0, 100.0):
            S.set_boost(boost)
            rec, cc = [], []
            for (rg, n, f, rho, R) in cells_for([1, 3]):
                for s in range(5):
                    res = run_trial(S, a.facts, n, f, rho, rg, "none", s, list(e04.TAB_METHODS), R)
                    for d in e04.TAB_METHODS:
                        rec.append(res[d]["recall"]); cc.append(res[d]["certified_correct_frac"])
            obj = float(np.nanmean(rec) + np.mean(cc))
            out[f"layer={layer}|boost={boost}"] = {"clean_recall": float(np.nanmean(rec)),
                                                 "certified_correct": float(np.mean(cc)), "objective": obj}
            print(f"layer={layer} boost={boost} recall={np.nanmean(rec):.3f} cert_correct={np.mean(cc):.3f} obj={obj:.3f}",
                  flush=True)
            if best is None or obj > best[0] + 1e-9:
                best = (obj, layer, boost)
    out["selected"] = {"layer": best[1], "boost": best[2]}
    out["base_lm_accuracy"] = S.base_accuracy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--trial_offset", type=int, default=0)
    ap.add_argument("--facts", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--boost", type=float, default=None, help="logits the memory adds to a stored token")
    ap.add_argument("--obs_rounds", default="1")
    ap.add_argument("--attacks", default=",".join(e04.ATTACKS))
    ap.add_argument("--defences", default=",".join(e04.DEFENCES))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = ROOT / "results" / "e10"; out.mkdir(parents=True, exist_ok=True)
    safe = a.model.replace("/", "_")
    tune_file = out / f"tuning_{safe}.json"
    layer, boost = a.layer, a.boost
    if not a.tune and (layer is None or boost is None) and tune_file.exists():
        sel = json.loads(tune_file.read_text())["selected"]
        layer = sel["layer"] if layer is None else layer
        boost = sel["boost"] if boost is None else boost
    if not (a.tune or a.smoke) and (layer is None or boost is None):
        sys.exit(f"no clean-tuned (layer, boost) for {a.model}: run with --tune first (Amendment 10)")
    S = Substrate(a.model, a.device, layer, 50.0 if boost is None else boost)
    print(f"model={a.model} layers={S.layer_grid} key_layer={S.layer} boost={S.boost} gain={S.gain:.2f} dk={S.dk} "
          f"pool={len(S.pool)} base_acc={S.base_accuracy()}", flush=True)
    if a.tune:
        res = tune(S, a)
        tune_file.write_text(json.dumps(res, indent=1))
        print("selected", res["selected"])
        return
    defences = a.defences.split(",")
    if a.smoke:
        cells, atks, trials = cells_for(None, True), ["none", "assert", "edge"], range(3)
    else:
        cells = cells_for([int(x) for x in a.obs_rounds.split(",")])
        atks, trials = a.attacks.split(","), range(a.trial_offset, a.trial_offset + a.trials)
    rows, t0 = [], time.time()
    for (rg, n, f, rho, R), at in itertools.product(cells, atks):
        for s in trials:
            res = run_trial(S, a.facts, n, f, rho, rg, at, s, defences, R)
            for d, r in res.items():
                rows.append({"regime": rg, "n": n, "f": f, "rho": rho, "R": R, "attack": at, "trial": s,
                             "defence": d, **r})
        print(f"{rg} n={n} f={f} rho={rho} R={R} {at}: {time.time() - t0:.0f}s", flush=True)
    name = "smoke" if a.smoke else f"e10{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    from scipy.stats import binom
    for (rg, n, f, rho, R) in cells:
        k_ = 2 * f + 1
        ms = [n - f] if rg == "dense" else list(range(k_, (n - f) // 2 + 1))
        p = 1 - (1 - rho) ** R
        h7 = float(np.mean([binom.sf(k_ - 1, m, p) / max(binom.sf(0, m, p), 1e-12) for m in ms]))
        for d in defences:
            rr = [r for r in rows if (r["regime"], r["f"], r["rho"], r["R"], r["defence"]) == (rg, f, rho, R, d)]
            att = [r for r in rr if r["attack"] != "none"]
            cl = [r for r in rr if r["attack"] == "none"]
            ax = [x for x in atks if x != "none"]
            e = {"clean_recall": float(np.nanmean([r["recall"] for r in cl])),
                 "worst_installed": float(max((np.mean([r["installed"] for r in att if r["attack"] == x]) for x in ax),
                                              default=float("nan"))),
                 "worst_amnesia": float(max((np.mean([r["amnesia"] for r in att if r["attack"] == x]) for x in ax),
                                            default=float("nan")))}
            if d in e04.TAB_METHODS:
                e.update({"H7_pred": h7, "violations": int(sum(r["violations"] for r in rr)),
                          "action_violations": int(sum(r["action_violations"] for r in rr)),
                          "certified_frac_clean": float(np.mean([r["certified_frac"] for r in cl])),
                          "certified_correct_clean": float(np.mean([r["certified_correct_frac"] for r in cl])),
                          "certified_frac_attacked": float(np.mean([r["certified_frac"] for r in att])) if att else float("nan")})
            summ[f"{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
            print(f"{rg:8} f={f} rho={rho} R={R} {d:15} " + " ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    tot_v = sum(v.get("violations", 0) for v in summ.values())
    tot_av = sum(v.get("action_violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "cert_violations": tot_v, "action_cert_violations": tot_av,
                                 "seconds": round(time.time() - t0, 1)}))
    log_run("e10_lm_retrofit", {**vars(a), "key_layer": S.layer, "read_boost": S.boost, "read_gain": S.gain},
            {"summary": summ, "cert_violations": tot_v, "action_violations": tot_av,
             "base_lm_accuracy": S.base_accuracy()},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
