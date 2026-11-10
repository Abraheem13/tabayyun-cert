"""Synthetic multi-source associative streams (the controlled testbed).

A world holds F facts (k_f, v_f).  n sources write into memory; the first
``n_adv`` sources are adversarial.  In every promotion round each honest
source independently observes each shared fact with probability ``rho``
and writes it with a persistent per-source bias of scale ``sigma``
(the honest spread Delta_H of Assumption 1).  A fraction ``unique_frac`` of
facts is observed by a single honest source only (the starvation regime of
Theorem 3(b) / Hypothesis 4).  ``dirichlet`` < inf makes observation rates
heterogeneous across sources (non-IID, Hypothesis 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WorldConfig:
    n_sources: int = 10
    n_adv: int = 2
    dk: int = 32
    dv: int = 32
    n_facts: int = 24
    rho: float = 0.6
    sigma: float = 0.05
    reps: int = 2              # writes per observation
    tau: int = 50
    unique_frac: float = 0.0
    dirichlet: float = float("inf")
    value_scale: float = 1.0
    support_min: int | None = None   # locality: each shared fact is known to a random subset of
    support_max: int | None = None   # m ~ U[support_min, support_max] honest sources (None = all)
    seed: int = 0


class FactWorld:
    def __init__(self, cfg: WorldConfig, device="cpu"):
        self.cfg, self.dev = cfg, device
        g = torch.Generator().manual_seed(cfg.seed)
        self.g = g
        F, dk, dv = cfg.n_facts, cfg.dk, cfg.dv
        if F <= dk:
            Q, _ = torch.linalg.qr(torch.randn(dk, dk, generator=g))
            self.keys = Q[:, :F].T.contiguous()
        else:
            k = torch.randn(F, dk, generator=g)
            self.keys = k / k.norm(dim=1, keepdim=True)
        v = torch.randn(F, dv, generator=g)
        self.values = cfg.value_scale * v / v.norm(dim=1, keepdim=True)
        n = cfg.n_sources
        self.adv = torch.arange(cfg.n_adv)
        self.honest = torch.arange(cfg.n_adv, n)
        self.bias = cfg.sigma * torch.randn(n, F, dv, generator=g) / dv ** 0.5
        # unique facts: owned by one honest source
        n_u = int(round(cfg.unique_frac * F))
        self.unique = torch.zeros(F, dtype=torch.bool)
        self.owner = torch.full((F,), -1, dtype=torch.long)
        if n_u:
            idx = torch.randperm(F, generator=g)[:n_u]
            self.unique[idx] = True
            self.owner[idx] = self.honest[torch.randint(len(self.honest), (n_u,), generator=g)]
        # non-IID observation rates
        if cfg.dirichlet == float("inf"):
            self.obs_p = torch.full((n, F), cfg.rho)
        else:
            d = torch.distributions.Dirichlet(torch.full((n,), cfg.dirichlet))
            torch.manual_seed(cfg.seed)
            w = d.sample((F,)).T                      # (n, F) columns sum to 1
            self.obs_p = (w * len(self.honest) * cfg.rho).clamp(max=1.0)
        self.support = torch.full((F,), len(self.honest))
        if cfg.support_min is not None:
            hi = min(cfg.support_max or len(self.honest), len(self.honest))
            for f in range(F):
                if self.unique[f]:
                    continue
                m = int(torch.randint(cfg.support_min, hi + 1, (1,), generator=g))
                keep = self.honest[torch.randperm(len(self.honest), generator=g)[:m]]
                mask = torch.ones(n, dtype=torch.bool)
                mask[keep] = False
                self.obs_p[mask, f] = 0.0
                self.support[f] = m
        for f in range(F):
            if self.unique[f]:
                self.obs_p[:, f] = 0.0
                self.obs_p[self.owner[f], f] = 1.0
        self.keys, self.values = self.keys.to(device), self.values.to(device)
        self.bias = self.bias.to(device)

    def honest_value(self, s: int, f: int) -> torch.Tensor:
        return self.values[f] + self.bias[s, f]

    def round_schedule(self) -> list[list[tuple[int, int]]]:
        """Per honest source, the list of facts written this round (with reps)."""
        cfg, g = self.cfg, self.g
        sched = [[] for _ in range(cfg.n_sources)]
        obs = torch.rand(cfg.n_sources, cfg.n_facts, generator=g) < self.obs_p
        for s in self.honest.tolist():
            fs = torch.nonzero(obs[s]).flatten().tolist() * cfg.reps
            perm = torch.randperm(len(fs), generator=g).tolist()
            sched[s] = [fs[i] for i in perm][: cfg.tau]
        return sched


def run_round(mem, world: FactWorld, attack=None, on_step=None) -> None:
    """Drive one promotion round (tau steps) of honest + adversarial writes.

    ``mem`` is any object with ``write(src, k, v)``; ``attack`` any object with
    ``writes(step, mem) -> list[(src, k, v)]``.
    """
    cfg = world.cfg
    sched = world.round_schedule()
    for t in range(cfg.tau):
        src, ks, vs = [], [], []
        for s in world.honest.tolist():
            if t < len(sched[s]):
                f = sched[s][t]
                src.append(s); ks.append(world.keys[f]); vs.append(world.honest_value(s, f))
        if attack is not None:
            for s, k, v in attack.writes(t, mem):
                src.append(s); ks.append(k); vs.append(v)
        if src:
            mem.write(torch.tensor(src, device=world.dev), torch.stack(ks), torch.stack(vs))
        else:
            mem.write(torch.zeros(0, dtype=torch.long, device=world.dev),
                      torch.zeros(0, cfg.dk, device=world.dev), torch.zeros(0, cfg.dv, device=world.dev))
        if on_step is not None:
            on_step(t)
