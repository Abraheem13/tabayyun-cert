"""Two-level (federated) Tabayyun-Cert (Sec. 6.4 / Thm 5).

N clients, each a TabayyunMemory over its own n sources.  Every server round:
  1. the server anchors global directions: a client-level key (a row of a
     client's own frozen basis) becomes a global direction once >= f_c + 1
     clients report it (so >= 1 honest client vouches for it);
  2. every client reports, along each global direction u, its promoted read
     M^P_c u and a per-coordinate local bound delta_c(u)_i = sum_v |<v,u>| E^c_{v,i};
  3. the server runs the SAME tabayyun gate across clients (k_c >= 2 f_c + 1);
  4. global memory M^G <- EMA towards the server values on promoted directions.

Theorem 5 (composition, proof in paper/proofs.md): with at most f_c Byzantine
clients (which may send arbitrary values AND arbitrary bounds),
  |M^G u - M~^G u|_i <= E^G_{u,i},  with per round
  e^G_{u,i} = e^server_{u,i} + max_{c} delta_c(u)_i     (max over ALL reports),
where M~^G uses, for each promoted direction, the mean of the honest
server-corroborators' *local honest references* M~^P_c u.  The max over all
reported bounds upper-bounds the max over honest clients, so a lying client
can only loosen (never invalidate) the certificate.
"""
from __future__ import annotations

from typing import Callable

import torch

from .certify import DirectionalDrift
from .gate import GateConfig, tabayyun_gate
from .tiered import TabayyunMemory, grow_anchored_basis


class FederatedTabayyun:
    def __init__(self, clients: list[TabayyunMemory], server_gate: GateConfig, alpha_G: float = 0.5,
                 anchor_cos: float = 0.98):
        self.clients = clients
        self.N = len(clients)
        self.gate = server_gate
        self.a = alpha_G
        dk, dv = clients[0].cfg.dk, clients[0].cfg.dv
        dev = clients[0].MP.device
        self.U = torch.zeros(0, dk, device=dev)
        self.MG = torch.zeros(dv, dk, device=dev)
        self.MG_ref = torch.zeros(dv, dk, device=dev)
        self.drift = DirectionalDrift(0, dv, "ema", alpha_G, 0.0, dev)
        self.anchor_cos = anchor_cos
        self.byzantine: list[int] = []          # indices of malicious clients
        self.server_hook: Callable | None = None  # (D, U) -> D for Byzantine rows
        self.last = None

    def sync(self):
        """One server round (call after every client has promoted)."""
        dk = self.U.shape[1]
        reports = [(c, u) for c, cl in enumerate(self.clients) for u in cl.basis()]
        self.U = grow_anchored_basis(self.U, [reports], self.N, self.gate.f + 1, dk, self.anchor_cos)
        K = self.U.shape[0]
        self.drift.resize(K)
        if K == 0:
            return None
        U = self.U
        D = torch.stack([U @ cl.MP.T for cl in self.clients])              # (N, K, dv) client promoted reads
        Dref = torch.stack([U @ cl.MP_ref.T for cl in self.clients])       # honest references (oracle)
        delta = torch.stack([(U @ cl.basis().T).abs() @ cl.drift.E if cl.basis().shape[0] else
                             torch.zeros(K, D.shape[-1], device=D.device) for cl in self.clients])  # (N,K,dv)
        if self.server_hook is not None and self.byzantine:
            D = self.server_hook(D, U)
            delta[self.byzantine] = 0.0                                      # Byzantine clients under-report
        out = tabayyun_gate(D, self.gate)
        e_g = out.runtime_err + delta.max(0).values                          # Theorem 5
        pm = out.promoted.to(D.dtype).unsqueeze(1)
        cur = U @ self.MG.T
        self.MG = self.MG + (self.a * pm * (out.values - cur)).T @ U
        # reference: honest server-corroborators' mean of their LOCAL honest references
        H = torch.ones(self.N, dtype=torch.bool, device=D.device)
        H[self.byzantine] = False
        CH = out.corroborators & H.unsqueeze(1)
        ref = (Dref * CH.unsqueeze(-1)).sum(0) / CH.sum(0).clamp(min=1).unsqueeze(-1)
        cur_r = U @ self.MG_ref.T
        self.MG_ref = self.MG_ref + (self.a * pm * (ref - cur_r)).T @ U
        self.drift.update(e_g, out.promoted)
        self.last = out
        return out

    def read(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self.MG.T

    @property
    def certificate(self) -> float:
        return self.drift.frobenius()

    def true_drift(self) -> float:
        return float((self.MG - self.MG_ref).norm())

    def query_certificate(self, q):
        return self.drift.query_bound(q, self.U)
