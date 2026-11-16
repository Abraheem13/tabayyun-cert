"""E05 - adaptive gradient attacker (PGD / BPDA) against Tier-B memory defences.

Threat model (strongest white-box): the adversary knows the model, the gate
and every honest source's memory, and sets the states of its own f sources
ARBITRARILY (unbounded) to make the trained MemLM emit its token v_adv at the
target keys.  For each defence the adversarial rows are optimised by Adam on
the Carlini-Wagner token loss through the defence's EXACT forward pass:
  * Tabayyun gate: the trimmed mean / median are piecewise-linear in the
    values (autograd through sort); membership and promotion are discrete,
    so a hinge term pulls adversarial rows into the cone (BPDA-style surrogate
    for the discrete corroborator test).
  * TrMean / median / GeoMed / Krum / RobustRAG: autograd through the exact op.
Restarts: (i) edge-of-range init, (ii) target-scaled init, (iii) random.
The best restart (lowest loss) is kept and evaluated with the exact forward.

Usage:
    python experiments/e05_adaptive.py --ckpt checkpoints/memlm_d128.pt --trials 50 --trial_offset 300
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
import e04_mqar_attack as E  # noqa: E402
from tabayyun.attacks import AgreeEdge, EdgeOfRange  # noqa: E402
from tabayyun.certify import action_certified  # noqa: E402
from tabayyun.data.mqar import segment  # noqa: E402
from tabayyun.gate import AGGREGATORS, GateConfig, tabayyun_gate  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402
from tabayyun.tiered import grow_anchored_basis  # noqa: E402

DEFS = ["tabayyun", "tabayyun_strict", "agg_trmean", "agg_median", "agg_geomed", "agg_krum", "robustrag"]


def cw_loss(logits, adv_tok):
    """sum over targets of (max_{j != adv} z_j - z_adv); < 0 means the attacker's token wins."""
    z_adv = logits.gather(1, adv_tok.view(-1, 1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, adv_tok.view(-1, 1), -1e9)
    return (other.max(1).values - z_adv)


def defence_read(d, D, U, q, f, n, k_gate):
    """Exact forward of defence d given per-source direction reads D (n,K,dv) along basis U (K,dk)."""
    if d == "tabayyun_w":
        gc = GateConfig(k=f + 1, f=f, rule="agree", eps=E.EPS_W, nu=1e-3, nu_rel=0.3)
        out = tabayyun_gate(D, gc)
        return q @ (out.values.T @ U).T, out
    if d.startswith("tabayyun"):
        gc = GateConfig(k=k_gate, f=f, gamma=0.3, nu=1e-3, nu_rel=None if d == "tabayyun" else 0.3)
        out = tabayyun_gate(D, gc)
        return q @ (out.values.T @ U).T, out
    if d == "robustrag":
        R = torch.einsum("nkv,mk->nmv", D, q)                             # D in identity basis here
        return R.median(0).values, None
    M = AGGREGATORS[d[4:]](D, f).T                                         # D in identity basis
    return q @ M.T, None


def attack_defence(model, d, D_hon, U, q_t, h_t, adv_tok, f, n, k_gate, honest_ids, ks, vs, steps, lr,
                   extra_inits=(), hinge_modes=(True,), keep_inits=False, select="loss"):
    """Optimise the f adversarial rows of D. Returns best adversarial rows.

    Defaults reproduce E05 (Amendment 5).  The union attacker (Amendment 12) adds ``extra_inits``
    (e.g. the adversary's real quarantine states under the static write attacks), runs the
    optimisation with and without the cone hinge (``hinge_modes``), and with ``keep_inits`` also
    keeps every un-optimised start, so it is never weaker than the static attacks it starts from."""
    K, dv = D_hon.shape[1], D_hon.shape[2]
    scale = float(D_hon[f:].norm(dim=-1).max().clamp(min=1e-3))
    inits = [EdgeOfRange(list(range(f)), honest_ids, ks, vs, nu=1e-3)(D_hon, U)[:f],
             ((U @ ks).unsqueeze(1) * vs.unsqueeze(0)).unsqueeze(0).expand(f, K, dv) * 3.0,
             torch.randn(f, K, dv).to(D_hon.device) * scale] + list(extra_inits)
    if d == "tabayyun_w":        # gate-specific start: just inside the agreement ball, towards the target
        inits.append(AgreeEdge(list(range(f)), honest_ids, ks, vs, eps=E.EPS_W, nu=1e-3)(D_hon, U)[:f])
    best, best_loss = None, (0, float("inf"))

    def exact_loss(A_):
        """Selection key, lower is better: summed CW loss (E05), or for select="count" the number
        of targets where the attacker's token wins first and the loss second."""
        with torch.no_grad():
            y_, _ = defence_read(d, torch.cat([A_, D_hon[f:]]), U, q_t, f, n, k_gate)
            cw = cw_loss(model.logits_from_read(h_t, y_), adv_tok)
            return (-int((cw < 0).sum()) if select == "count" else 0, float(cw.sum()))

    if keep_inits:
        for init in inits:
            fl = exact_loss(init)
            if fl < best_loss:
                best_loss, best = fl, init.detach().clone()
    for hinge, init in itertools.product(hinge_modes, inits):
        A = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([A], lr=lr * scale)
        for _ in range(steps):
            D = torch.cat([A, D_hon[f:]])
            y, out = defence_read(d, D, U, q_t, f, n, k_gate)
            loss = cw_loss(model.logits_from_read(h_t, y), adv_tok).clamp(min=-1.0).sum()
            if out is not None and hinge and d == "tabayyun_w":
                # BPDA surrogate for the agreement gate: stay inside the ball of an honest witness
                Hn = D_hon[f:]
                act = Hn.norm(dim=-1) >= 1e-3                              # (nh, K)
                mean_h = (Hn * act.unsqueeze(-1)).sum(0) / act.sum(0).clamp(min=1).unsqueeze(-1)
                r = E.EPS_W * Hn.norm(dim=-1).max(0).values                # (K,)
                dist = (A - mean_h.unsqueeze(0)).norm(dim=-1)              # (f, K)
                loss = loss + (torch.relu(dist - 0.9 * r.unsqueeze(0)) / r.clamp(min=1e-6).unsqueeze(0)).sum()
            elif out is not None and hinge:   # BPDA surrogate: keep adversarial rows active and inside the cone
                med = D.detach()[f:].median(0).values                      # honest-majority reference
                cos = torch.nn.functional.cosine_similarity(A, med.unsqueeze(0), dim=-1)
                loss = loss + torch.relu(0.35 - cos).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        fl = exact_loss(A.detach())
        if fl < best_loss:
            best_loss, best = fl, A.detach().clone()
    return best


UNION_DEFS = ["tabayyun", "tabayyun_strict", "agg_trmean", "tabayyun_w"]


def run(model, V, F, n, f, rho, regime, R, seed, steps, lr, device="cpu", union=False):
    keys, vals, adv_vals, targets, hidx, g = E.build_trial(V, F, n, f, rho, regime, seed, R)
    segs = [torch.zeros(0, dtype=torch.long)] * f + [
        segment(keys, vals, torch.tensor(hidx[s], dtype=torch.long), g) if hidx[s] else torch.zeros(0, dtype=torch.long)
        for s in range(f, n)]
    with torch.no_grad():
        Ms, written = E.per_source_memories(model, segs, device)
        h_q, q = E.query_features(model, keys, V, device)
        ks, vs = E.kv_vectors(model, int(keys[targets[0]]), int(adv_vals[targets[0]]), device)
    # adversary is a writer of the target keys (it asserts them), so they get anchored as for E04 attacks
    adv_written = [(s, E.kv_vectors(model, int(keys[t]), int(adv_vals[t]), device)[0]) for s in range(f) for t in targets]
    if not union:
        return attack_all(model, Ms, written, adv_written, q, h_q, vals, adv_vals, targets, ks, vs, f, n, steps, lr)
    # Amendment 13: the adversary's REAL quarantines under the static write attacks as extra starts
    adv_states = []
    with torch.no_grad():
        for i, at in enumerate(("assert", "assert_x4")):
            g2 = torch.Generator().manual_seed(seed * 7919 + i + 1)
            Madv, w_adv = E.per_source_memories(model, E.adversarial_segments(keys, vals, adv_vals, targets, f, at, g2, F),
                                                device)
            adv_states.append(Madv)
            adv_written = adv_written + w_adv          # camouflage keys count as written keys, as in E04
    return attack_all(model, Ms, written, adv_written, q, h_q, vals, adv_vals, targets, ks, vs, f, n, steps, lr,
                      defs=UNION_DEFS, adv_states=adv_states)


def attack_all(model, Ms, written, adv_written, q, h_q, vals, adv_vals, targets, ks, vs, f, n, steps, lr,
               certify_fn=action_certified, defs=None, adv_states=None):
    """Run the adaptive attacker against every defence in DEFS.  Substrate-agnostic: ``model`` only
    needs ``logits_from_read(h, y)`` and ``readout_matrix()`` (MemLM in E05, a frozen-LM shim in E11)."""
    dev = Ms.device
    k_gate = 2 * f + 1
    honest_ids = list(range(f, n))
    Ua = grow_anchored_basis(torch.zeros(0, Ms.shape[-1], device=dev), [written + adv_written], n, f + 1, Ms.shape[-1])
    I = torch.eye(Ms.shape[-1], device=dev)
    tq, th = q[targets], h_q[targets]
    adv_tok = adv_vals[targets].to(dev)
    true_tok = vals[targets].to(dev)
    res = {}
    union = adv_states is not None           # Amendment 12 union attacker (see attack_defence)
    for d in (defs or DEFS):
        U = Ua if d.startswith("tabayyun") else I
        if U.shape[0] == 0:
            res[d] = {"installed": 0.0, "edge_installed": 0.0}
            continue
        D_hon = torch.einsum("nvk,uk->nuv", Ms, U)
        if union:
            statics = [torch.einsum("nvk,uk->nuv", Ma, U) for Ma in adv_states]
            A = attack_defence(model, d, D_hon, U, tq, th, adv_tok, f, n, k_gate, honest_ids, ks, vs, steps, lr,
                               extra_inits=statics, keep_inits=True, select="count",
                               hinge_modes=(True, False) if d.startswith("tabayyun") else (True,))
        else:
            A = attack_defence(model, d, D_hon, U, tq, th, adv_tok, f, n, k_gate, honest_ids, ks, vs, steps, lr)
        with torch.no_grad():
            D = torch.cat([A, D_hon[f:]])
            y, out = defence_read(d, D, U, tq, f, n, k_gate)
            pred = model.logits_from_read(th, y).argmax(-1)
            De = EdgeOfRange(list(range(f)), honest_ids, ks, vs, nu=1e-3)(D_hon, U)
            ye, _ = defence_read(d, De, U, tq, f, n, k_gate)
            pe = model.logits_from_read(th, ye).argmax(-1)
            r = {"installed": float((pred == adv_tok).float().mean()),
                 "amnesia": float((pred != true_tok).float().mean()),
                 "edge_installed": float((pe == adv_tok).float().mean())}
            if union:      # what the plain write attacks achieve on the same trial (no optimisation)
                st = [defence_read(d, torch.cat([A_s, D_hon[f:]]), U, tq, f, n, k_gate)[0] for A_s in statics]
                r["static_installed"] = float(max((model.logits_from_read(th, y_s).argmax(-1) == adv_tok)
                                                  .float().mean() for y_s in st))
            if out is not None:
                # certificate soundness under the adaptive attack (oracle reference = honest-corroborator mean)
                H = torch.zeros(n, dtype=torch.bool, device=dev); H[f:] = True
                CH = out.corroborators & H.unsqueeze(1)
                ref = (D * CH.unsqueeze(-1)).sum(0) / CH.sum(0).clamp(min=1).unsqueeze(-1)
                ref = ref * out.promoted.unsqueeze(1).to(ref.dtype)
                MP, MPr = out.values.T @ U, ref.T @ U
                drift_q = ((tq @ U.T).abs() @ out.runtime_err).norm(dim=-1)
                true_q = ((MP - MPr) @ tq.T).T.norm(dim=-1)
                logits = model.logits_from_read(th, y)
                cert = certify_fn(logits, model.readout_matrix(), drift_q)
                ref_pred = model.logits_from_read(th, tq @ MPr.T).argmax(-1)
                r["violations"] = int((true_q > drift_q * (1 + 1e-4) + 1e-5).sum())
                r["action_violations"] = int(((logits.argmax(-1) != ref_pred) & cert).sum())
                r["certified_targets"] = float(cert.float().mean())
                # certify-or-abstain serving: a target counts as installed only if its token is certified
                r["served_installed"] = float(((logits.argmax(-1) == adv_tok) & cert).float().mean())
                r["served_correct"] = float(((logits.argmax(-1) == true_tok) & cert).float().mean())
            res[d] = r
    return res


_MODEL = None


def _init(ckpt, eps_w=None, union_defs=None):
    global _MODEL
    if eps_w is not None:
        E.EPS_W = eps_w
    if union_defs:                      # spawn-safe (Windows): workers re-import this module
        UNION_DEFS[:] = union_defs
    torch.set_num_threads(1)
    torch.manual_seed(0)
    _MODEL = E.load(ckpt, "cpu")
    for p in _MODEL.parameters():
        p.requires_grad_(False)


def _job(job):
    (rg, n, f, rho, R), s, steps, lr = job[:4]
    out = run(_MODEL, _MODEL.cfg.vocab, 16, n, f, rho, rg, R, s, steps, lr, union=len(job) > 4 and job[4])
    return [{"regime": rg, "f": f, "rho": rho, "R": R, "trial": s, "defence": d, **r} for d, r in out.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--trial_offset", type=int, default=300)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--union", action="store_true", help="Amendment 13 union attacker (see attack_defence)")
    ap.add_argument("--eps_w", type=float, default=None, help="override the clean-tuned eps of tabayyun_w (control runs)")
    ap.add_argument("--defences", default=None, help="comma list (default: all for --union)")
    ap.add_argument("--cells", default=None, help="comma list of cell indices (default: all 16)")
    a = ap.parse_args()
    cells = [(rg, n, f, rho, R) for rg, n, fs in [("dense", 10, (1, 2)), ("locality", 20, (2, 3))]
             for f in fs for rho in (0.4, 0.7) for R in (1, 3)]
    trials = range(a.trial_offset, a.trial_offset + a.trials)
    if a.defences:
        UNION_DEFS[:] = a.defences.split(",")
    if a.cells:
        cells = [cells[int(i)] for i in a.cells.split(",")]
    if a.smoke:
        cells, trials, a.steps = cells[:2] + cells[-2:], range(2), 30
    rows, t0 = [], time.time()
    jobs = [(c, s, a.steps, a.lr, a.union) for c, s in itertools.product(cells, trials)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.ckpt, a.eps_w, list(UNION_DEFS))) as ex:
        for i, rr in enumerate(ex.map(_job, jobs)):
            rows.extend(rr)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "results" / "e05"; out.mkdir(parents=True, exist_ok=True)
    name = ("smoke_union" if a.union else "smoke") if a.smoke else f"e05{('_' + a.tag) if a.tag else ''}"
    defs = UNION_DEFS if a.union else DEFS
    (out / f"{name}.json").write_text(json.dumps(rows))
    summ = {}
    for (rg, n, f, rho, R) in cells:
        for d in defs:
            rr = [r for r in rows if (r["regime"], r["f"], r["rho"], r["R"], r["defence"]) == (rg, f, rho, R, d)]
            e = {"pgd_installed": float(np.mean([r["installed"] for r in rr])),
                 "amnesia_mean": float(np.mean([r["amnesia"] for r in rr])),
                 "edge_installed": float(np.mean([r["edge_installed"] for r in rr]))}
            if a.union:
                e["static_installed"] = float(np.mean([r["static_installed"] for r in rr]))
                if any("served_installed" in r for r in rr):
                    e["served_installed"] = float(np.mean([r["served_installed"] for r in rr]))
                    e["served_correct"] = float(np.mean([r["served_correct"] for r in rr]))
            if any("violations" in r for r in rr):
                e["violations"] = int(sum(r.get("violations", 0) + r.get("action_violations", 0) for r in rr))
            summ[f"{rg}|f={f}|rho={rho}|R={R}|{d}"] = e
            print(f"{rg:8} f={f} rho={rho} R={R} {d:15} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    (out / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
    viol = sum(v.get("violations", 0) for v in summ.values())
    print("SUMMARY", json.dumps({"rows": len(rows), "violations": viol, "seconds": round(time.time() - t0, 1)}))
    log_run("e05_adaptive", vars(a), {"summary": summ, "violations": viol},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
