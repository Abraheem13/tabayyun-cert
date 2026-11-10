"""E04 - Tier B: poisoning a trained memory LM (MemLM on MQAR), token level.

For each trial: F facts; n sources (first f adversarial). Honest source s asserts
the facts in its support (dense: all; locality: m ~ U[k, (n-f)//2] honest
sources per fact) with probability rho each, as a segment [k v k v ...].
Adversarial sources assert the target facts with a WRONG value token (and
camouflage with some true facts).  The trained model's own memory layer is run
  * over the concatenation of all segments (ungated Titans), or
  * on every segment separately (per-source quarantines M_s),
and the defences combine the M_s.  Queries are all F keys; the predicted token
is argmax of logits = W_U(h_q + W_O y), y = memory read.

Metrics (Amendment 3): recall = token accuracy on benign facts; installed =
prediction equals the attacker's token at a target; amnesia = prediction differs
from the true value at a target.  Tabayyun additionally reports the fraction of
queries whose predicted TOKEN is certified (Thm 3a, exact pairwise form,
W = W_U W_O) and checks soundness against the honest-corroborator reference.

Usage:
    python experiments/e04_mqar_attack.py --ckpt checkpoints/memlm_d128.pt --smoke
    python experiments/e04_mqar_attack.py --ckpt checkpoints/memlm_d128.pt --trials 100 --trial_offset 200
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun.attacks import EdgeOfRange, Unbounded  # noqa: E402
from tabayyun.certify import action_certified  # noqa: E402
from tabayyun.data.mqar import sample_facts, segment  # noqa: E402
from tabayyun.gate import AGGREGATORS, GateConfig, tabayyun_gate  # noqa: E402
from tabayyun.models import MemLM, MemLMConfig  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402
from tabayyun.tiered import grow_anchored_basis  # noqa: E402

NU_REL = None      # set by --nu_rel (v2.3); tuned on CLEAN trials only
GAMMA = 0.3
DEFENCES = ["titans", "agg_mean", "agg_trmean", "agg_median", "agg_krum", "agg_geomed",
            "robustrag", "smsr", "tma_corr", "tab_nocorr", "tabayyun", "tabayyun_strict"]
TAB_METHODS = ("tabayyun", "tabayyun_strict")
NU_REL_STRICT = 0.3   # secondary variant (Amendment 4): strict k-witness semantics
ATTACKS = ["none", "assert", "assert_x4", "edge", "unbounded"]


def load(ckpt, device):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    m = MemLM(MemLMConfig(**d["cfg"])).to(device)
    m.load_state_dict(d["state"])
    m.eval()
    return m


@torch.no_grad()
def per_source_memories(model, segs, device):
    """Run the memory layer on every segment separately. Returns M (n,dv,dk) and
    the written keys [(source, key_vec)] at value positions."""
    n = len(segs)
    L = max(1, max(len(s) for s in segs))
    tok = torch.zeros(n, L, dtype=torch.long, device=device)
    wm = torch.zeros(n, L, dtype=torch.bool, device=device)
    for i, s in enumerate(segs):
        if len(s):
            tok[i, :len(s)] = s.to(device)
            wm[i, 1:len(s):2] = True
    h = model.encode(tok)
    k, v, q, th, al, et = model.mem_inputs(h)
    _, M, _ = model.scan(k, v, q, th, al, et, wm)
    written = [(i, k[i, t]) for i in range(n) for t in range(1, len(segs[i]), 2)]
    return M, written


@torch.no_grad()
def concat_memory(model, segs, device, g):
    order = torch.randperm(len(segs), generator=g).tolist()
    seq = torch.cat([segs[i] for i in order if len(segs[i])]).to(device)[None]
    wm = torch.zeros_like(seq, dtype=torch.bool)
    wm[0, 1::2] = True
    h = model.encode(seq)
    k, v, q, th, al, et = model.mem_inputs(h)
    _, M, _ = model.scan(k, v, q, th, al, et, wm)
    return M[0]


@torch.no_grad()
def query_features(model, keys, V, device):
    seq = torch.cat([torch.tensor([V // 2]), keys]).to(device)[None]
    h = model.encode(seq)[0, 1:]
    _, _, q, _, _, _ = model.mem_inputs(h)
    return h, q


@torch.no_grad()
def kv_vectors(model, key_tok, val_tok, device):
    """Memory key / value vectors the model writes for the pair (key_tok, val_tok)."""
    h = model.encode(torch.tensor([[key_tok, val_tok]], device=device))
    k, v, _, _, _, _ = model.mem_inputs(h)
    return k[0, 1], v[0, 1]


def build_trial(V, F, n, f, rho, regime, seed, rounds=1):
    g = torch.Generator().manual_seed(seed)
    keys, vals = sample_facts(F, V, g)
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
    for t in targets:   # attacker's token: a different value token
        adv_vals[t] = V // 2 + (int(vals[t]) - V // 2 + 1 + int(torch.randint(0, V // 2 - 1, (1,), generator=g))) % (V // 2)
    honest_idx = {s: [] for s in honest}
    for _r in range(rounds):           # a fact observed in r rounds is asserted r times
        for j in range(F):
            for s in support[j]:
                if torch.rand(1, generator=g).item() < rho:
                    honest_idx[s].append(j)
    for t in targets:            # targets must be known to >= 1 honest source to be attackable
        if not any(t in honest_idx[s] for s in honest):
            honest_idx[support[t][0]].append(t)
    return keys, vals, adv_vals, targets, honest_idx, g


def adversarial_segments(keys, vals, adv_vals, targets, f, attack, g, F):
    segs = []
    for _ in range(f):
        camo = torch.randperm(F, generator=g)[: max(1, F // 3)].tolist()
        camo = [j for j in camo if j not in targets]
        reps = 4 if attack == "assert_x4" else 1
        idx = camo + targets * reps
        kk = torch.cat([keys[camo], keys[targets].repeat(reps)])
        vv = torch.cat([vals[camo], adv_vals[targets].repeat(reps)])
        perm = torch.randperm(len(idx), generator=g)
        segs.append(torch.stack([kk[perm], vv[perm]], 1).flatten())
    return segs


def predict(model, h_q, y):
    return model.logits_from_read(h_q, y)


def run_trial(model, V, F, n, f, rho, regime, attack, seed, device, defences, rounds=1):
    keys, vals, adv_vals, targets, honest_idx, g = build_trial(V, F, n, f, rho, regime, seed, rounds)
    honest_segs = [segment(keys, vals, torch.tensor(honest_idx[s], dtype=torch.long), g)
                   if honest_idx[s] else torch.zeros(0, dtype=torch.long) for s in range(f, n)]
    if attack == "none":
        adv_segs = [torch.zeros(0, dtype=torch.long) for _ in range(f)]
    else:
        adv_segs = adversarial_segments(keys, vals, adv_vals, targets, f, attack, g, F)
    segs = adv_segs + honest_segs
    Ms, written = per_source_memories(model, segs, device)                 # (n, dv, dk)
    h_q, q = query_features(model, keys, V, device)                        # (F, d), (F, dk)
    k_gate = 2 * f + 1
    hook = None
    if attack in ("edge", "unbounded"):
        ks, vs = kv_vectors(model, int(keys[targets[0]]), int(adv_vals[targets[0]]), device)
        hook = (EdgeOfRange(list(range(f)), list(range(f, n)), ks, vs, nu=1e-3) if attack == "edge"
                else Unbounded(list(range(f)), ks, vs, scale=20.0))
    I = torch.eye(Ms.shape[-1], device=device)
    res = {}
    benign = [j for j in range(F) if j not in targets and any(j in honest_idx[s] for s in honest_idx)]
    for d in defences:
        cert = None
        if d == "titans":
            Mg = concat_memory(model, segs, device, g)
            y = q @ Mg.T
        elif d.startswith("agg_") or d in ("robustrag", "smsr"):
            D = Ms.transpose(1, 2)                                         # (n, dk, dv) rows = M_s e_i
            if hook is not None:
                D = hook(D, I)
            if d.startswith("agg_"):
                Mg = AGGREGATORS[d[4:]](D, f).T
                y = q @ Mg.T
            else:
                R = torch.einsum("nkv,mk->nmv", D, q)                     # per-source reads (n, F, dv)
                if d == "robustrag":
                    y = R.median(0).values
                else:
                    gg = torch.Generator().manual_seed(seed)
                    s_ = max(1, round(0.3 * n))
                    subs = [torch.randperm(n, generator=gg)[:s_] for _ in range(21)]
                    y = torch.stack([R[sb].mean(0) for sb in subs]).median(0).values
        else:   # tabayyun / tab_nocorr / tma_corr: anchored basis + gate
            U = grow_anchored_basis(torch.zeros(0, Ms.shape[-1], device=device),
                                    [written], n, f + 1, Ms.shape[-1])
            if U.shape[0] == 0:
                y = torch.zeros(F, Ms.shape[1], device=device)
            else:
                D = torch.einsum("nvk,uk->nuv", Ms, U)                     # (n, K, dv)
                if hook is not None:
                    D = hook(D, U)
                gc = {"tabayyun": GateConfig(k=k_gate, f=f, gamma=GAMMA, nu=1e-3, nu_rel=NU_REL),
                      "tabayyun_strict": GateConfig(k=k_gate, f=f, gamma=GAMMA, nu=1e-3, nu_rel=NU_REL_STRICT),
                      "tab_nocorr": GateConfig(k=k_gate, f=f, corroborate=False),
                      "tma_corr": GateConfig(k=k_gate, f=f, nu=1e-3, nu_rel=NU_REL, cone=False, trim=False)}[d]
                out = tabayyun_gate(D, gc)
                MP = out.values.T @ U                                      # single promotion round, alpha_P = 1
                y = q @ MP.T
                if d in TAB_METHODS:
                    # reference: honest-corroborator mean under the same decisions (oracle, evaluation only)
                    H = torch.zeros(n, dtype=torch.bool, device=device); H[f:] = True
                    CH = out.corroborators & H.unsqueeze(1)
                    ref = (D * CH.unsqueeze(-1)).sum(0) / CH.sum(0).clamp(min=1).unsqueeze(-1)
                    ref = ref * out.promoted.unsqueeze(1).to(ref.dtype)
                    MP_ref = ref.T @ U
                    w = (q @ U.T).abs()                                    # (F, K)
                    drift_q = (w @ out.runtime_err).norm(dim=-1)
                    true_q = ((MP - MP_ref) @ q.T).T.norm(dim=-1)
                    logits = predict(model, h_q, y)
                    certified = action_certified(logits, model.readout_matrix(), drift_q)
                    ref_pred = predict(model, h_q, q @ MP_ref.T).argmax(-1)
                    cert = {"violations": int((true_q > drift_q * (1 + 1e-4) + 1e-5).sum()),
                            "action_violations": int(((logits.argmax(-1) != ref_pred) & certified).sum()),
                            "certified_frac": float(certified.float().mean()),
                            "certified_frac_benign": float(certified[benign].float().mean()) if benign else math.nan,
                            "certified_correct_frac": float((certified & (logits.argmax(-1) == vals.to(device))).float().mean()),
                            "n_directions": int(U.shape[0])}
        pred = predict(model, h_q, y).argmax(-1).cpu()
        r = {"recall": float((pred[benign] == vals[benign]).float().mean()) if benign else math.nan,
             "installed": float(np.mean([int(pred[t] == adv_vals[t]) for t in targets])) if attack != "none" else None,
             "amnesia": float(np.mean([int(pred[t] != vals[t]) for t in targets]))}
        if cert is not None:
            r.update(cert)
        res[d] = r
    return res


def main():
    global NU_REL, GAMMA, DEFENCES
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--trial_offset", type=int, default=0, help="confirmatory: 200+ (Amendment 3)")
    ap.add_argument("--facts", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nu_rel", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--attacks", default=",".join(ATTACKS))
    ap.add_argument("--defences", default=",".join(DEFENCES))
    ap.add_argument("--obs_rounds", default="1", help="comma list of observation rounds R (Amendment 4: 1,3)")
    a = ap.parse_args()
    NU_REL, GAMMA = a.nu_rel, a.gamma
    DEFENCES = a.defences.split(",")
    torch.set_grad_enabled(False)
    model = load(a.ckpt, a.device)
    V = model.cfg.vocab
    if a.smoke:
        cells = [("dense", 10, 2, 0.7, 1), ("locality", 20, 2, 0.7, 3)]
        atks, trials = ["none", "assert", "edge"], range(3)
    else:
        cells = [(rg, n, f, rho, R) for rg, n, fs in [("dense", 10, (1, 2)), ("locality", 20, (2, 3))]
                 for f in fs for rho in (0.4, 0.7) for R in [int(x) for x in a.obs_rounds.split(",")]]
        atks, trials = a.attacks.split(","), range(a.trial_offset, a.trial_offset + a.trials)
    rows, t0 = [], time.time()
    for (rg, n, f, rho, R), at in itertools.product(cells, atks):
        for s in trials:
            res = run_trial(model, V, a.facts, n, f, rho, rg, at, s, a.device, DEFENCES, R)
            for d, r in res.items():
                rows.append({"regime": rg, "n": n, "f": f, "rho": rho, "R": R, "attack": at, "trial": s,
                             "defence": d, **r})
        print(f"{rg} n={n} f={f} rho={rho} R={R} {at}: {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "results" / "e04"; out.mkdir(parents=True, exist_ok=True)
    name = "smoke" if a.smoke else f"e04{('_' + a.tag) if a.tag else ''}"
    (out / f"{name}.json").write_text(json.dumps(rows))
    # summary table
    summ = {}
    for (rg, n, f, rho, R) in cells:
        k_ = 2 * f + 1
        ms = [n - f] if rg == "dense" else list(range(k_, (n - f) // 2 + 1))
        p = 1 - (1 - rho) ** R
        from scipy.stats import binom
        h7 = float(np.mean([binom.sf(k_ - 1, m, p) / max(binom.sf(0, m, p), 1e-12) for m in ms]))
        for d in DEFENCES:
            rr = [r for r in rows if r["regime"] == rg and r["f"] == f and r["rho"] == rho and r["R"] == R
                  and r["defence"] == d]
            att = [r for r in rr if r["attack"] != "none"]
            cl = [r for r in rr if r["attack"] == "none"]
            ax = [x for x in atks if x != "none"]
            worst_inst = max((np.mean([r["installed"] for r in att if r["attack"] == x]) for x in ax), default=float("nan"))
            worst_amn = max((np.mean([r["amnesia"] for r in att if r["attack"] == x]) for x in ax), default=float("nan"))
            e = {"clean_recall": float(np.nanmean([r["recall"] for r in cl])), "worst_installed": float(worst_inst),
                 "worst_amnesia": float(worst_amn)}
            if d in TAB_METHODS:
                e["H7_pred"] = h7
                e["violations"] = int(sum(r["violations"] for r in rr))
                e["action_violations"] = int(sum(r["action_violations"] for r in rr))
                e["certified_frac_clean"] = float(np.mean([r["certified_frac"] for r in cl]))
                e["certified_correct_clean"] = float(np.mean([r["certified_correct_frac"] for r in cl]))
                e["certified_frac_attacked"] = float(np.mean([r["certified_frac"] for r in att])) if att else float("nan")
            summ[f"{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
            print(f"{rg:8} f={f} rho={rho} R={R} {d:15} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    tot_v = sum(v.get("violations", 0) for v in summ.values())
    tot_av = sum(v.get("action_violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "cert_violations": tot_v, "action_cert_violations": tot_av,
                                 "seconds": round(time.time() - t0, 1)}))
    log_run("e04_mqar_attack", vars(a), {"summary": summ, "cert_violations": tot_v, "action_violations": tot_av},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
