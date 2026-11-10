"""MQAR (multi-query associative recall, Zoology protocol) in a multi-source form.

Vocabulary: 0 = pad, keys in [1, V/2), values in [V/2, V).
A *fact* is a (key, value) token pair.  A *source segment* is a sequence of
the facts that source asserts, in random order: [k1 v1 k2 v2 ...].  A
*query section* lists keys; at each query position the model must emit the
key's value token (next-token prediction at the key's own position).

Training batches concatenate 1..max_segments honest segments (facts may repeat
across segments, never conflict), followed by a query section; the loss is
only on query positions.  Writes happen at the *value* position of every
asserted fact (one write per assertion): this matches the threat model, where
each write is one source asserting one (key, value) fact, and it makes MQAR
learnable for the single-memory-layer model within ~1k steps (Day 6 probe:
96% vs a 32% plateau when every context position writes).
"""
from __future__ import annotations

import torch


def key_range(V):
    return 1, V // 2


def value_range(V):
    return V // 2, V


def sample_facts(n_facts: int, V: int, g: torch.Generator):
    klo, khi = key_range(V)
    vlo, vhi = value_range(V)
    keys = klo + torch.randperm(khi - klo, generator=g)[:n_facts]
    vals = vlo + torch.randint(0, vhi - vlo, (n_facts,), generator=g)
    return keys, vals


def segment(keys, vals, idx, g):
    """Tokens of one source segment asserting facts idx (random order)."""
    order = idx[torch.randperm(len(idx), generator=g)]
    seg = torch.stack([keys[order], vals[order]], 1).flatten()
    return seg


def train_batch(B: int, V: int, g: torch.Generator, n_pairs=(8, 32), max_segments=6, n_queries=16,
                device="cpu"):
    """Returns tokens (B,L), write_mask (B,L), targets (B,L) with -100 off-query."""
    seqs, wms, tgts = [], [], []
    for _ in range(B):
        F = int(torch.randint(n_pairs[0], n_pairs[1] + 1, (1,), generator=g))
        keys, vals = sample_facts(F, V, g)
        nseg = int(torch.randint(1, max_segments + 1, (1,), generator=g))
        segs = []
        covered = torch.zeros(F, dtype=torch.bool)
        for _s in range(nseg):
            rho = float(torch.empty(1).uniform_(0.3, 1.0, generator=g))
            idx = torch.nonzero(torch.rand(F, generator=g) < rho).flatten()
            if len(idx) == 0:
                idx = torch.randint(0, F, (1,), generator=g)
            covered[idx] = True
            segs.append(segment(keys, vals, idx, g))
        ctx = torch.cat(segs)
        cov = torch.nonzero(covered).flatten()
        qi = cov[torch.randint(0, len(cov), (n_queries,), generator=g)]
        qtok = keys[qi]
        seq = torch.cat([ctx, qtok])
        tgt = torch.full_like(seq, -100)
        tgt[len(ctx):] = vals[qi]
        wm = torch.zeros_like(seq, dtype=torch.bool)
        wm[1: len(ctx): 2] = True          # one write per asserted fact, at its value position
        seqs.append(seq); wms.append(wm); tgts.append(tgt)
    L = max(len(s) for s in seqs)
    pad = lambda xs, v: torch.stack([torch.cat([x, torch.full((L - len(x),), v, dtype=x.dtype)]) for x in xs])  # noqa: E731
    # left-align context, right-pad: padded positions neither write nor are scored
    return pad(seqs, 0).to(device), pad(wms, False).to(device), pad(tgts, -100).to(device)
