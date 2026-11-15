"""Attack suite (Sec. 9.3).

Write-level attacks implement ``writes(step, mem) -> list[(src, k, v)]`` and
go through the normal write path.  White-box promotion-level attacks
implement ``__call__(D, U) -> D`` and are installed as
``TabayyunMemory.delta_hook``: they rewrite the rows of the adversarial
sources directly, i.e. the adversary can put *any* state into its own
quarantines (the strongest write-level adversary, since it controls every
write of its sources).
"""
from __future__ import annotations

import torch


class TargetedWrite:
    """f sources write (k*, v*) at every step (insertion or flip).

    ``scale`` > 1 gives the large-norm single-shot variant when ``steps`` = [0].
    """

    def __init__(self, adv_ids, k_star, v_star, scale=1.0, steps=None):
        self.adv, self.k, self.v, self.scale, self.steps = list(adv_ids), k_star, v_star, scale, steps

    def writes(self, t, mem):
        if self.steps is not None and t not in self.steps:
            return []
        return [(s, self.k, self.scale * self.v) for s in self.adv]


class Rayn:
    """'rayn' attack: repeated low-norm writes, each moving the read at k*
    by at most ``eps`` towards v* (white-box w.r.t. the current memory).
    Designed to slip under per-write norm filters and to ride momentum."""

    def __init__(self, adv_ids, k_star, v_star, eps=0.05, every=1):
        self.adv, self.k, self.v, self.eps, self.every = list(adv_ids), k_star, v_star, eps, every

    def writes(self, t, mem):
        if t % self.every:
            return []
        out = []
        for s in self.adv:
            cur = _read_as(mem, s, self.k)
            d = self.v - cur
            step = d * min(1.0, self.eps / (float(d.norm()) + 1e-12))
            out.append((s, self.k, cur + step))
        return out


def _read_as(mem, s, k):
    if hasattr(mem, "Q"):                                # quarantine of source s
        n = mem.Q.B
        q = torch.zeros(n, mem.Q.dk, device=k.device)
        q[s] = k
        return mem.Q.read(q)[s]
    return mem.read(k.unsqueeze(0)).squeeze(0)


# ------------------------------------------------- white-box promotion hooks
class EdgeOfRange:
    """Adaptive corroboration-forging attack (Sec. 9.3 item 5), omniscient.

    For each direction u, the adversarial rows are placed at the coordinate-
    wise edge of the honest corroborators' range in the direction of the
    attacker's target t_u (= v* (u.k*) for an insertion/flip), while staying
    inside the cosine cone of the honest median so that they are counted as
    corroborators.  Directions with no honest support are pushed with full
    force (Theorem 1 says this has no effect).
    """

    def __init__(self, adv_ids, honest_ids, k_star, v_star, nu=1e-3, push=1.0):
        self.adv = torch.as_tensor(adv_ids)
        self.h = torch.as_tensor(honest_ids)
        self.k, self.v, self.nu, self.push = k_star, v_star, nu, push

    def __call__(self, D, U):
        D = D.clone()
        target = (U @ self.k).unsqueeze(1) * self.v.unsqueeze(0)       # (K, dv)
        Hd = D[self.h]                                                 # (nh, K, dv)
        active = Hd.norm(dim=-1) >= self.nu                            # (nh, K)
        inf = torch.tensor(float("inf"))
        hmax = torch.where(active.unsqueeze(-1), Hd, -inf).max(0).values
        hmin = torch.where(active.unsqueeze(-1), Hd, inf).min(0).values
        has = active.any(0)                                            # (K,)
        mean = (Hd * active.unsqueeze(-1)).sum(0) / active.sum(0).clamp(min=1).unsqueeze(-1)
        edge = torch.where(target > mean, hmax, hmin)
        val = torch.where(has.unsqueeze(-1), mean + self.push * (edge - mean), target)
        D[self.adv] = val.unsqueeze(0).expand(len(self.adv), *val.shape)
        return D


class AgreeEdge:
    """Adaptive attack on the agreement gate (omniscient): the adversarial rows sit at distance
    rho * eps * (the gate's own scale) from the honest mean, pushed towards the attacker's target, so that they
    are accepted as agreeing witnesses while shifting the promoted value as far as allowed.
    Directions with no honest support get the full target (no effect, by Theorem 1)."""

    def __init__(self, adv_ids, honest_ids, k_star, v_star, eps, nu=1e-3, rho=0.95):
        self.adv = torch.as_tensor(adv_ids)
        self.h = torch.as_tensor(honest_ids)
        self.k, self.v, self.eps, self.nu, self.rho = k_star, v_star, eps, nu, rho

    def __call__(self, D, U):
        D = D.clone()
        target = (U @ self.k).unsqueeze(1) * self.v.unsqueeze(0)       # (K, dv)
        Hd = D[self.h]
        nrm = Hd.norm(dim=-1)                                          # (nh, K)
        active = nrm >= self.nu
        cnt = active.sum(0).clamp(min=1)
        mean = (Hd * active.unsqueeze(-1)).sum(0) / cnt.unsqueeze(-1)
        scale = (nrm * active).sum(0) / cnt                            # mean honest norm
        d = target - mean
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # The gate's tolerance is eps * (f+1)-th largest active norm, which depends on the adversarial
        # rows themselves (moving towards the target changes their norm).  Solve for the largest
        # admissible step by fixed-point iteration on the gate's own scale.
        f = len(self.adv)
        hn = nrm * active                                              # (nh, K), 0 where inactive
        step = self.eps * scale                                        # (K,)
        for _ in range(8):
            an = (mean + step.unsqueeze(-1) * d).norm(dim=-1)          # adversarial norm, (K,)
            alln = torch.cat([hn, an.unsqueeze(0).expand(f, -1)])
            s_gate = torch.topk(alln, min(f + 1, alln.shape[0]), dim=0).values[-1]
            step = self.rho * self.eps * s_gate
        val = torch.where(active.any(0).unsqueeze(-1), mean + step.unsqueeze(-1) * d, target)
        D[self.adv] = val.unsqueeze(0).expand(len(self.adv), *val.shape)
        return D


class Unbounded:
    """Arbitrary rows (e.g. huge values towards v*): tests that trimming and
    the cone test, not norm filtering, provide the guarantee."""

    def __init__(self, adv_ids, k_star, v_star, scale=100.0):
        self.adv, self.k, self.v, self.scale = torch.as_tensor(adv_ids), k_star, v_star, scale

    def __call__(self, D, U):
        D = D.clone()
        target = (U @ self.k).unsqueeze(1) * self.v.unsqueeze(0)
        D[self.adv] = self.scale * target
        return D


class MedianShift:
    """Liveness attack: rotate the median so that honest sources fail the
    cosine test (denial of promotion rather than insertion)."""

    def __init__(self, adv_ids, scale=1.0, seed=0):
        self.adv, self.scale = torch.as_tensor(adv_ids), scale
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, D, U):
        D = D.clone()
        m = D.mean(0)                                                  # (K, dv)
        r = torch.randn(m.shape, generator=self.g).to(D.device)
        r = r - (r * m).sum(-1, keepdim=True) / (m.pow(2).sum(-1, keepdim=True) + 1e-12) * m
        r = r / (r.norm(dim=-1, keepdim=True) + 1e-12) * (m.norm(dim=-1, keepdim=True) * self.scale)
        D[self.adv] = (-m + r).unsqueeze(0).expand(len(self.adv), *m.shape)
        return D
