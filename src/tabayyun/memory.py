"""Titans-style neural test-time memories (batched).

Every memory object holds B independent memories (B = number of sources for a
quarantine tier, or B = 1 for a single shared memory).  All of them follow the
Titans / Miras update (Eq. 2 of the proposal):

    g_t = grad_M || M(k_t) - v_t ||_2^2           (attentional bias, Eq. 1)
    g_t <- clip so that || theta * g_t || <= theta * G
    S_t = eta * S_{t-1} - theta * g_t               (momentum / "surprise")
    M_t = (1 - alpha) * M_{t-1} + S_t               (forget gate)

Gates are constant here (the theory regime).  Data-dependent gates are used
by the learned models in ``tabayyun.models``.

Two memory families:
  * ``LinearMemory``  M in R^{dv x dk}; the regime in which every certificate
    is exact.
  * ``MLPMemory``     2-layer residual MLP, the Titans LMM regime.

Both expose the same interface: ``write``, ``read``, ``snapshot``,
``read_snapshot``, ``reset_momentum``, ``state_norm``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Gates:
    theta: float = 0.5   # inner learning rate
    eta: float = 0.0     # momentum
    alpha: float = 0.0   # forget gate
    clip: float | None = None  # G: per-write gradient-norm clip (None = off)


def _clip_scale(gnorm: torch.Tensor, clip: float | None) -> torch.Tensor:
    if clip is None:
        return torch.ones_like(gnorm)
    return torch.clamp(clip / (gnorm + 1e-12), max=1.0)


class LinearMemory:
    """B independent linear memories M_b in R^{dv x dk}."""

    def __init__(self, B: int, dk: int, dv: int, gates: Gates,
                 device: str | torch.device = "cpu", dtype=None):
        self.B, self.dk, self.dv, self.g = B, dk, dv, gates
        dtype = dtype or torch.get_default_dtype()
        self.M = torch.zeros(B, dv, dk, device=device, dtype=dtype)
        self.S = torch.zeros_like(self.M)
        self._snap = torch.zeros_like(self.M)

    # ------------------------------------------------------------------ io
    def read(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, ..., dk) -> (B, ..., dv)."""
        return torch.einsum("bvk,b...k->b...v", self.M, q)

    def read_snapshot(self, q: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bvk,b...k->b...v", self._snap, q)

    def grad(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-memory gradient of ||M k - v||^2. k: (B,dk), v: (B,dv)."""
        err = torch.einsum("bvk,bk->bv", self.M, k) - v
        return 2.0 * torch.einsum("bv,bk->bvk", err, k)

    def write(self, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None,
              theta_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """One Titans step on every memory whose ``mask`` entry is True.

        Memories with mask False do not move (each source runs on its own
        clock).  Returns the applied (clipped) surprise norms ``||theta g||``.
        """
        g = self.grad(k, v)
        gnorm = g.flatten(1).norm(dim=1)
        g = g * _clip_scale(gnorm, self.g.clip).view(-1, 1, 1)
        th = self.g.theta * (theta_scale if torch.is_tensor(theta_scale)
                             else torch.full((self.B,), float(theta_scale),
                                             device=self.M.device))
        S_new = self.g.eta * self.S - th.view(-1, 1, 1) * g
        M_new = (1.0 - self.g.alpha) * self.M + S_new
        if mask is None:
            self.S, self.M = S_new, M_new
        else:
            m = mask.view(-1, 1, 1).to(self.M.dtype)
            self.S = m * S_new + (1 - m) * self.S
            self.M = m * M_new + (1 - m) * self.M
        return (th.view(-1, 1, 1) * g).flatten(1).norm(dim=1)

    # ------------------------------------------------------------ helpers
    def snapshot(self) -> None:
        self._snap = self.M.clone()

    def reset_to_snapshot(self) -> None:
        self.M = self._snap.clone()
        self.S.zero_()

    def reset(self) -> None:
        self.M.zero_(); self.S.zero_(); self._snap.zero_()

    def decay(self, factor: float) -> None:
        self.M.mul_(factor); self.S.mul_(factor)

    def reset_momentum(self) -> None:
        self.S.zero_()

    def state_norm(self) -> torch.Tensor:
        return self.M.flatten(1).norm(dim=1)


class MLPMemory:
    """B independent 2-layer residual MLP memories (Titans LMM regime).

    M(k) = W2 silu(W1 k) + P k,   P a fixed (non-learned) skip so that a fresh
    memory reads ~0 for the residual part only when W2 = 0.
    """

    def __init__(self, B: int, dk: int, dv: int, gates: Gates, hidden: int = 64,
                 device: str | torch.device = "cpu", dtype=None,
                 seed: int = 0):
        self.B, self.dk, self.dv, self.h, self.g = B, dk, dv, hidden, gates
        dtype = dtype or torch.get_default_dtype()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        W1 = torch.randn(hidden, dk, generator=gen, dtype=torch.float64) / dk ** 0.5
        self.W1 = W1.to(device, dtype).expand(B, hidden, dk).clone()
        self.W2 = torch.zeros(B, dv, hidden, device=device, dtype=dtype)
        self.params = [self.W1, self.W2]
        self.mom = [torch.zeros_like(p) for p in self.params]
        self._snap = [p.clone() for p in self.params]

    @staticmethod
    def _fwd(W1, W2, q):
        h = torch.nn.functional.silu(torch.einsum("bhk,b...k->b...h", W1, q))
        return torch.einsum("bvh,b...h->b...v", W2, h)

    def read(self, q):
        return self._fwd(self.W1, self.W2, q)

    def read_snapshot(self, q):
        return self._fwd(self._snap[0], self._snap[1], q)

    def write(self, k, v, mask=None, theta_scale=1.0):
        W1 = self.W1.detach().requires_grad_(True)
        W2 = self.W2.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = ((self._fwd(W1, W2, k) - v) ** 2).sum()
            g1, g2 = torch.autograd.grad(loss, [W1, W2])
        gnorm = torch.sqrt(g1.flatten(1).pow(2).sum(1) + g2.flatten(1).pow(2).sum(1))
        sc = _clip_scale(gnorm, self.g.clip)
        th = self.g.theta * (theta_scale if torch.is_tensor(theta_scale)
                             else torch.full((self.B,), float(theta_scale),
                                             device=self.W1.device))
        out = []
        for i, (p, g) in enumerate(zip(self.params, (g1, g2))):
            shape = (-1,) + (1,) * (p.dim() - 1)
            step = (th * sc).view(shape) * g
            S_new = self.g.eta * self.mom[i] - step
            P_new = (1.0 - self.g.alpha) * p + S_new
            if mask is None:
                self.mom[i], self.params[i] = S_new, P_new
            else:
                m = mask.view(shape).to(p.dtype)
                self.mom[i] = m * S_new + (1 - m) * self.mom[i]
                self.params[i] = m * P_new + (1 - m) * p
            out.append(step.flatten(1).pow(2).sum(1))
        self.W1, self.W2 = self.params
        return torch.sqrt(sum(out))

    def snapshot(self):
        self._snap = [p.clone() for p in self.params]

    def reset_to_snapshot(self):
        self.params = [p.clone() for p in self._snap]
        self.W1, self.W2 = self.params
        for m in self.mom:
            m.zero_()

    def reset(self):
        self.W2.zero_()
        for m in self.mom:
            m.zero_()
        self.snapshot()

    def decay(self, factor):
        # decay only the learned read-out so the feature map is preserved
        self.W2.mul_(factor); self.mom[1].mul_(factor)
        self.params[1] = self.W2

    def reset_momentum(self):
        for m in self.mom:
            m.zero_()

    def state_norm(self):
        return self.W2.flatten(1).norm(dim=1)


def make_memory(kind: str, B: int, dk: int, dv: int, gates: Gates, **kw):
    if kind == "linear":
        return LinearMemory(B, dk, dv, gates, **{k: v for k, v in kw.items() if k in ("device", "dtype")})
    if kind == "mlp":
        return MLPMemory(B, dk, dv, gates, **kw)
    raise ValueError(kind)
