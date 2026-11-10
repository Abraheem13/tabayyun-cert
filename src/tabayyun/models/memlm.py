"""MemLM: a small Titans-style memory language model for MQAR (Tier B).

Architecture (all cross-token mixing happens in ONE neural-memory layer, so
that per-source quarantine of that layer isolates sources completely):

    x -> Embedding -> causal ShortConv(width 2) -> + MLP  (position-wise)    h_t
    k_t = norm(W_K h_t), v_t = W_V h_t, q_t = norm(W_Q h_t)
    data-dependent gates theta_t, alpha_t, eta_t = sigmoid(w . h_t + b) * max
    Titans linear memory:  S_t = eta_t S_{t-1} - theta_t grad,  M_t = (1-alpha_t) M_{t-1} + S_t
    read  y_t = M_{t-1} q_t
    logits_t = W_U (h_t + W_o y_t)          <- linear in y_t (no final norm)

Because logits are affine in the memory read y, Theorem 3(a) gives an EXACT
pairwise certificate for the predicted token with W = W_U W_o.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemLMConfig:
    vocab: int = 256
    d_model: int = 128
    dk: int = 64
    dv: int = 64
    mlp_mult: int = 2
    theta_max: float = 1.0
    alpha_max: float = 0.1
    eta_max: float = 0.9


class MemLM(nn.Module):
    def __init__(self, cfg: MemLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = nn.Embedding(cfg.vocab, d)
        # full (not depthwise) width-2 conv: h_t = A x_{t-1} + B x_t keeps the previous token
        # (the key, at a value position) linearly separable from the current one
        self.conv = nn.Conv1d(d, d, kernel_size=2, bias=True)
        self.norm = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_mult * d), nn.GELU(), nn.Linear(cfg.mlp_mult * d, d))
        self.WK = nn.Linear(d, cfg.dk, bias=False)
        self.WV = nn.Linear(d, cfg.dv, bias=False)
        self.WQ = nn.Linear(d, cfg.dk, bias=False)
        self.gates = nn.Linear(d, 3)
        self.WO = nn.Linear(cfg.dv, d, bias=False)
        self.WU = nn.Linear(d, cfg.vocab, bias=True)

    # ------------------------------------------------------------ encoders
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Position-wise features with a width-2 causal conv. tokens: (B, L) -> (B, L, d)."""
        x = self.emb(tokens)
        xc = F.pad(x.transpose(1, 2), (1, 0))
        h = x + self.conv(xc).transpose(1, 2)
        return h + self.mlp(self.norm(h))

    def mem_inputs(self, h: torch.Tensor):
        c = self.cfg
        k = F.normalize(self.WK(h), dim=-1)
        q = F.normalize(self.WQ(h), dim=-1)
        v = self.WV(h)
        g = torch.sigmoid(self.gates(h))
        return k, v, q, g[..., 0] * c.theta_max, g[..., 1] * c.alpha_max, g[..., 2] * c.eta_max

    # --------------------------------------------------------- memory scan
    @staticmethod
    def scan(k, v, q, theta, alpha, eta, write_mask, M0=None, S0=None):
        """Sequential Titans linear-memory scan.
        k,q: (B,L,dk) v: (B,L,dv) gates: (B,L) write_mask: (B,L) bool.
        Returns reads y (B,L,dv) (read before write) and final (M, S)."""
        B, L, dk = k.shape
        dv = v.shape[-1]
        M = torch.zeros(B, dv, dk, device=k.device, dtype=k.dtype) if M0 is None else M0
        S = torch.zeros_like(M) if S0 is None else S0
        ys = []
        for t in range(L):
            ys.append(torch.einsum("bvk,bk->bv", M, q[:, t]))
            err = torch.einsum("bvk,bk->bv", M, k[:, t]) - v[:, t]
            grad = 2 * torch.einsum("bv,bk->bvk", err, k[:, t])
            w = write_mask[:, t].to(k.dtype).view(B, 1, 1)
            S_new = eta[:, t].view(B, 1, 1) * S - theta[:, t].view(B, 1, 1) * grad
            M_new = (1 - alpha[:, t].view(B, 1, 1)) * M + S_new
            S = w * S_new + (1 - w) * S
            M = w * M_new + (1 - w) * M
        return torch.stack(ys, 1), M, S

    def logits_from_read(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.WU(h + self.WO(y))

    def forward(self, tokens, write_mask):
        h = self.encode(tokens)
        k, v, q, th, al, et = self.mem_inputs(h)
        y, _, _ = self.scan(k, v, q, th, al, et, write_mask)
        return self.logits_from_read(h, y)

    # ------------------------------------------------------- certification
    def readout_matrix(self) -> torch.Tensor:
        """W such that logits = W y + (affine in h): W = W_U W_O, shape (vocab, dv)."""
        return self.WU.weight @ self.WO.weight
