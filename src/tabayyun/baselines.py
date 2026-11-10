"""Single-client baselines that share the ``write`` / ``read`` interface.

* ``SharedMemory``     ungated Titans memory: every source writes into one
                       memory.  Options reproduce clipping-only, OWASP-style
                       TTL (strong forget) and DP-noise memories.
* ``RobustAggMemory``  per-source quarantines + robust aggregation of the
                       whole quarantine states every tau steps (mean = FedAvg
                       analogue, trmean, median, krum, geomed, cclip).  This is
                       "Titans + Krum / trimmed mean applied to memory deltas"
                       from Sec. 9.4, and isolates the value of the
                       corroboration gate and the per-direction certificate.
"""
from __future__ import annotations

import torch

from .gate import AGGREGATORS, centered_clipping
from .memory import Gates, make_memory


class SharedMemory:
    def __init__(self, dk, dv, gates: Gates, kind="linear", dp_sigma: float = 0.0,
                 device="cpu", seed=0, **kw):
        self.mem = make_memory(kind, 1, dk, dv, gates, device=device, **kw)
        self.dp_sigma = dp_sigma
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.t = 0

    def write(self, src, k, v):
        for i in range(len(src)):           # writes inside one step applied in order
            self.mem.write(k[i:i + 1], v[i:i + 1])
            if self.dp_sigma > 0 and hasattr(self.mem, "M"):
                noise = torch.randn(self.mem.M.shape, generator=self.gen) * self.dp_sigma
                self.mem.M += noise.to(self.mem.M.device)
        self.t += 1

    def read(self, q, src=None, high_risk=True):
        return self.mem.read(q.unsqueeze(0)).squeeze(0)


class RobustAggMemory:
    def __init__(self, n_sources, dk, dv, gates: Gates, agg="mean", f=1, tau=50,
                 alpha_P=0.5, kind="linear", device="cpu", cclip_tau=1.0, basis="identity", **kw):
        self.Q = make_memory(kind, n_sources, dk, dv, gates, device=device, **kw)
        self.MP = torch.zeros(dv, dk, device=device)
        self.n, self.dk, self.dv = n_sources, dk, dv
        self.agg, self.f, self.tau, self.a = agg, f, tau, alpha_P
        self.cclip_tau = cclip_tau
        self.t = 0
        self.basis = basis
        self.U = torch.eye(dk, device=device) if basis == "identity" else torch.zeros(0, dk, device=device)
        self._keys, self._hist = [[] for _ in range(n_sources)], []
        self.delta_hook = None          # same white-box adversary as for TabayyunMemory

    def write(self, src, k, v):
        if len(src):
            Kf = torch.zeros(self.n, self.dk, device=self.MP.device)
            Vf = torch.zeros(self.n, self.dv, device=self.MP.device)
            mask = torch.zeros(self.n, dtype=torch.bool, device=self.MP.device)
            Kf[src], Vf[src], mask[src] = k, v, True
            self.Q.write(Kf, Vf, mask=mask)
            if self.basis == "anchored":
                for s_, kk in zip(src.tolist(), k):
                    self._keys[s_].append(kk.detach())
        self.t += 1
        if self.t % self.tau == 0:
            self.promote()

    def promote(self):
        if self.basis == "anchored":
            # fairness ablation A11: same key-anchored basis as Tabayyun (f+1 writers)
            from .tiered import grow_anchored_basis
            self._hist = (self._hist + [[(s_, k) for s_, ks in enumerate(self._keys) for k in ks]])[-5:]
            self._keys = [[] for _ in range(self.n)]
            self.U = grow_anchored_basis(self.U, self._hist, self.n, self.f + 1, self.dk)
            if self.U.shape[0] == 0:
                return
        Uq = self.U.unsqueeze(0).expand(self.n, *self.U.shape)
        D = self.Q.read(Uq)                                      # (n, dk, dv): states
        if self.delta_hook is not None:
            D = self.delta_hook(D, self.U)
        if self.agg == "cclip":
            ref = (self.U @ self.MP.T)
            val = centered_clipping(D, ref, self.cclip_tau)
        else:
            val = AGGREGATORS[self.agg](D, self.f)
        cur = self.U @ self.MP.T
        self.MP = self.MP + (self.a * (val - cur)).T @ self.U

    def read(self, q, src=None, high_risk=True):
        return q @ self.MP.T


# ---------------------------------------------------------------------------
# 2026 agent-memory defences, ported to parametric (gradient-written) memory.
# Each keeps the defining mechanism of the original and is given the same
# omniscient adversary as Tabayyun (via ``delta_hook`` on the per-source reads).
# ---------------------------------------------------------------------------
class HuberSharedMemory(SharedMemory):
    """Miras 'Yaad'-style memory: Huber attentional bias, i.e. the per-coordinate
    residual is clipped at delta before the gradient step (robust to outliers
    in a single write, but still a single shared tier)."""

    def __init__(self, dk, dv, gates: Gates, delta=0.1, device="cpu", **kw):
        super().__init__(dk, dv, gates, device=device, **kw)
        self.delta = delta

    def write(self, src, k, v):
        m = self.mem
        for i in range(len(src)):
            kk, vv = k[i:i + 1], v[i:i + 1]
            err = torch.einsum("bvk,bk->bv", m.M, kk) - vv
            g = 2.0 * torch.einsum("bv,bk->bvk", err.clamp(-self.delta, self.delta), kk)
            m.S = m.g.eta * m.S - m.g.theta * g
            m.M = (1 - m.g.alpha) * m.M + m.S
        self.t += 1


class _QuarantineReader:
    """Per-source quarantines read at query time (no promoted tier)."""

    def __init__(self, n_sources, dk, dv, gates: Gates, device="cpu"):
        self.Q = make_memory("linear", n_sources, dk, dv, gates, device=device)
        self.n, self.dk, self.dv = n_sources, dk, dv
        self.delta_hook = None
        self.t = 0

    def write(self, src, k, v):
        if len(src):
            dev = self.Q.M.device
            Kf = torch.zeros(self.n, self.dk, device=dev)
            Vf = torch.zeros(self.n, self.dv, device=dev)
            mask = torch.zeros(self.n, dtype=torch.bool, device=dev)
            Kf[src], Vf[src], mask[src] = k, v, True
            self.Q.write(Kf, Vf, mask=mask)
        self.t += 1

    def _reads(self, q):
        R = self.Q.read(q.unsqueeze(0).expand(self.n, *q.shape))          # (n, m, dv)
        if self.delta_hook is not None:                                   # adversary sets its rows
            qn = torch.nn.functional.normalize(q, dim=-1)
            R = self.delta_hook(R, qn) * q.norm(dim=-1, keepdim=True).unsqueeze(0)
        return R


class SMSRMemory(_QuarantineReader):
    """SMSR-style (arXiv 2606.12703) randomised source ablation + voting: for
    each query, R random subsets of s sources each produce a mean read; the
    answer is the coordinate-wise median ("vote") of the subset answers."""

    def __init__(self, n_sources, dk, dv, gates, R=21, frac=0.3, seed=0, device="cpu"):
        super().__init__(n_sources, dk, dv, gates, device)
        g = torch.Generator().manual_seed(seed)
        s = max(1, int(round(frac * n_sources)))
        self.subsets = torch.stack([torch.randperm(n_sources, generator=g)[:s] for _ in range(R)])

    def read(self, q, src=None, high_risk=True):
        Rd = self._reads(q)                                               # (n, m, dv)
        answers = torch.stack([Rd[sub].mean(0) for sub in self.subsets])  # (R, m, dv)
        return answers.median(0).values


class IsolateAggregateMemory(_QuarantineReader):
    """RobustRAG-style isolate-then-aggregate: every source answers in isolation
    and the answers are robustly aggregated at query time (coordinate median)."""

    def read(self, q, src=None, high_risk=True):
        return self._reads(q).median(0).values
