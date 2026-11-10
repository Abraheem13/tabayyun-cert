"""E03 - train MemLM on multi-segment MQAR (clean data only).

Usage:
    python experiments/e03_mqar_train.py --smoke                 # CPU sanity check
    python experiments/e03_mqar_train.py --device cuda --steps 8000 --tag lmm_d128
Saves checkpoints/<tag>.pt (git-ignored) and results/e03/<tag>.json (curve + final accuracy).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabayyun.data.mqar import train_batch  # noqa: E402
from tabayyun.models import MemLM, MemLMConfig  # noqa: E402
from tabayyun.registry import log_run  # noqa: E402


@torch.no_grad()
def evaluate(model, V, device, n_batches=8, B=128, seed=12345, **kw):
    g = torch.Generator().manual_seed(seed)
    correct = total = 0
    model.eval()
    for _ in range(n_batches):
        tok, wm, tgt = train_batch(B, V, g, device=device, **kw)
        pred = model(tok, wm).argmax(-1)
        m = tgt != -100
        correct += int((pred[m] == tgt[m]).sum()); total += int(m.sum())
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--max_pairs", type=int, default=24)
    ap.add_argument("--max_segments", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="memlm")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.d_model, a.dk, a.max_pairs, a.max_segments, a.tag = 300, 32, 64, 32, 8, 3, "smoke"
    torch.manual_seed(a.seed)
    cfg = MemLMConfig(vocab=a.vocab, d_model=a.d_model, dk=a.dk, dv=a.dk)
    model = MemLM(cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (1.0 if s < 0.8 * a.steps else
                                                   max(0.05, (a.steps - s) / (0.2 * a.steps))))
    g = torch.Generator().manual_seed(a.seed + 1)
    kw = dict(n_pairs=(max(2, a.max_pairs // 4), a.max_pairs), max_segments=a.max_segments)
    curve, t0 = [], time.time()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.2f}M  device {a.device}", flush=True)
    for step in range(1, a.steps + 1):
        # curriculum: the number of facts grows from 4 to max_pairs over the first 40% of steps
        frac = min(1.0, step / (0.4 * a.steps))
        hi = int(round(4 + frac * (a.max_pairs - 4)))
        ckw = dict(n_pairs=(2, max(4, hi)), max_segments=max(1, int(round(1 + frac * (a.max_segments - 1)))))
        tok, wm, tgt = train_batch(a.batch, a.vocab, g, device=a.device, **ckw)
        logits = model(tok, wm)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, a.vocab), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % (50 if a.smoke else 250) == 0 or step == a.steps:
            acc = evaluate(model, a.vocab, a.device, n_batches=2 if a.smoke else 4, **kw)
            curve.append({"step": step, "loss": float(loss), "acc": acc, "sec": round(time.time() - t0, 1)})
            print(json.dumps(curve[-1]), flush=True)
    final = evaluate(model, a.vocab, a.device, n_batches=4 if a.smoke else 16, **kw)
    ck = ROOT / "checkpoints"; ck.mkdir(exist_ok=True)
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "args": vars(a)}, ck / f"{a.tag}.pt")
    out = ROOT / "results" / "e03"; out.mkdir(parents=True, exist_ok=True)
    rec = {"final_acc": final, "params": n_params, "curve": curve, "args": vars(a)}
    (out / f"{a.tag}.json").write_text(json.dumps(rec, indent=1))
    print("SUMMARY", json.dumps({"final_acc": final, "params": n_params, "sec": round(time.time() - t0, 1)}))
    log_run("e03_mqar_train", vars(a), {"final_acc": final, "params": n_params},
            out="results/smoke.jsonl" if a.smoke else "results/registry.jsonl", smoke=a.smoke)


if __name__ == "__main__":
    main()
