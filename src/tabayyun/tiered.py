"""Tabayyun-Cert memory (one client): per-source quarantine + promoted tier.

    write  : source s writes (k, v)        -> only M_s^Q moves
    read   : y = M^P q + lam * 1[low-risk] * M_{s}^Q q        (Eq. 4)
    promote: every tau steps, tabayyun_gate over the quarantines (Eqs. 5-8)

Two promotion rules:

* ``ema`` (v2, default).  The gate runs on quarantine *reads* along each
  basis direction, D[s,u] = M_s^Q(u), and every promoted direction is moved
  towards the corroborated value by a convex step:
        M^P <- M^P + sum_{u promoted} alpha_P (value_u - M^P u) u^T
  Unpromoted directions decay by (1 - eps_P).  The error recursion is a
  convex combination, so the drift certificate does NOT grow like 1/alpha_P
  (Theorem 2', docs/proposal_v2.md).
* ``additive`` (v1 of the proposal, kept for the ablation).  The gate runs
  on read-*deltas* since the last promotion and
        M^P <- (1 - alpha_P) M^P + sum_u value_u u^T.

The promoted tier is linear and momentum-free by construction, also when
the quarantine tier is a deep MLP memory ("deep quarantine, linear certified
tier"), so the certificates are exact for any quarantine architecture.
The basis must be fixed for the per-direction certificate; data-driven
bases are therefore frozen after the first promotion round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .certify import DirectionalDrift
from .gate import GateConfig, GateOutput, coord_trimmed_mean, tabayyun_gate
from .memory import Gates, make_memory


@dataclass
class TabayyunConfig:
    n_sources: int
    dk: int
    dv: int
    quarantine_kind: str = "linear"           # linear | mlp
    q_gates: Gates = field(default_factory=lambda: Gates(theta=0.5, eta=0.0, alpha=0.02, clip=None))
    rule: str = "ema"                         # ema (v2) | additive (v1)
    alpha_P: float = 0.5                      # promoted-tier step (ema) or decay (additive)
    eps_P: float = 0.0                        # decay of unpromoted directions (ema)
    tau: int = 50                             # promotion period
    gate: GateConfig = field(default_factory=GateConfig)
    basis: str = "identity"                   # identity | anchored | keys_svd | robust_keys_svd
    anchor_cos: float = 0.98                  # anchored: keys this close are the same direction
    anchor_min_resid: float = 0.3             # anchored: min new component to add a direction
    anchor_window: int = 5                    # anchored: rounds of key history kept
    anchor_min_writers: str = "k"             # "k" (v2.1, Day 3) | "f+1" (v2.2): distinct writers to anchor
    K: int | None = None                      # number of directions (default dk)
    after_promotion: str = "keep"             # keep | reset | decay   (quarantine)
    post_decay: float = 0.5
    lam_low: float = 1.0                      # own-quarantine read weight, low-risk queries
    mlp_hidden: int = 64
    device: str = "cpu"


def grow_anchored_basis(U: torch.Tensor, hist, n_sources: int, need: int, K: int,
                        anchor_cos: float = 0.98, min_resid: float = 0.3) -> torch.Tensor:
    """Grow a frozen orthonormal basis with keys written by >= `need` distinct
    sources in the key history `hist` (list of rounds of (source, key)).
    Shared by TabayyunMemory and the anchored robust-aggregation baselines."""
    pool = [x for rnd in hist for x in rnd]
    if not pool or U.shape[0] >= K:
        return U
    src = torch.tensor([p[0] for p in pool], device=U.device)
    X = torch.nn.functional.normalize(torch.stack([p[1] for p in pool]), dim=1)
    close = (X @ X.T).abs() >= anchor_cos                                # (P,P)
    onehot = torch.nn.functional.one_hot(src, n_sources).to(X.dtype)     # (P,n)
    n_src = ((close.to(X.dtype) @ onehot) > 0).sum(1)                    # distinct writers per key
    for i in torch.argsort(n_src, descending=True).tolist():
        if int(n_src[i]) < need or U.shape[0] >= K:
            break
        members = X[close[i]]
        sign = torch.sign(members @ X[i]).unsqueeze(1)
        c = torch.nn.functional.normalize((members * sign).median(0).values, dim=0)
        if U.shape[0]:
            if (U @ c).abs().max() >= anchor_cos:
                continue
            r = c - U.T @ (U @ c)
        else:
            r = c
        if r.norm() >= min_resid:
            U = torch.cat([U, (r / r.norm()).unsqueeze(0)])
    return U


class TabayyunMemory:
    def __init__(self, cfg: TabayyunConfig):
        self.cfg = cfg
        n, dk, dv = cfg.n_sources, cfg.dk, cfg.dv
        kw = {"device": cfg.device}
        if cfg.quarantine_kind == "mlp":
            kw["hidden"] = cfg.mlp_hidden
        self.Q = make_memory(cfg.quarantine_kind, n, dk, dv, cfg.q_gates, **kw)
        self.MP = torch.zeros(dv, dk, device=cfg.device)
        self.t = 0
        self.rounds = 0
        self.K = cfg.K or dk
        self.U: torch.Tensor | None = None
        if cfg.basis == "anchored":
            self.U = torch.zeros(0, dk, device=cfg.device)
            self._hist: list[list[tuple[int, torch.Tensor]]] = []
        K0 = 0 if cfg.basis == "anchored" else self.K
        self.drift = DirectionalDrift(K0, dv, cfg.rule, cfg.alpha_P, cfg.eps_P, cfg.device)
        self._keys: list[list[torch.Tensor]] = [[] for _ in range(n)]
        self.last: GateOutput | None = None
        # white-box adversary: may rewrite the rows of the sources it controls
        self.delta_hook: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        # evaluation-only oracle: ids of honest sources, to build the reference M~^P
        self.oracle_honest: torch.Tensor | None = None
        self.MP_ref = torch.zeros_like(self.MP)
        self.Q.snapshot()

    # ----------------------------------------------------------------- io
    def write(self, src: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """One time step. src: (m,) distinct source ids; k: (m,dk); v: (m,dv)."""
        n, dev = self.cfg.n_sources, self.MP.device
        Kf = torch.zeros(n, self.cfg.dk, device=dev)
        Vf = torch.zeros(n, self.cfg.dv, device=dev)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        if len(src):
            Kf[src], Vf[src], mask[src] = k, v, True
            self.Q.write(Kf, Vf, mask=mask)
            if self.cfg.basis == "anchored" or (self.cfg.basis != "identity" and self.U is None):
                for s, kk in zip(src.tolist(), k):
                    self._keys[s].append(kk.detach())
        self.t += 1
        if self.t % self.cfg.tau == 0:
            self.promote()

    def read(self, q: torch.Tensor, src: torch.Tensor | None = None,
             high_risk: bool | torch.Tensor = True) -> torch.Tensor:
        """q: (m, dk). Low-risk queries additionally see the asking source's quarantine."""
        y = q @ self.MP.T
        if src is None or self.cfg.lam_low == 0:
            return y
        m = q.shape[0]
        hr = high_risk if torch.is_tensor(high_risk) else torch.full((m,), bool(high_risk), device=q.device)
        n = self.cfg.n_sources
        Qq = torch.zeros(n, m, self.cfg.dk, device=q.device)
        ar = torch.arange(m, device=q.device)
        Qq[src, ar] = q
        own = self.Q.read(Qq)[src, ar]
        return y + self.cfg.lam_low * (~hr).to(y.dtype).unsqueeze(1) * own

    # ------------------------------------------------------------ promotion
    def basis(self) -> torch.Tensor:
        if self.cfg.basis == "anchored":
            return self._grow_anchored()
        if self.U is not None:
            return self.U
        dk, dev = self.cfg.dk, self.MP.device
        if self.cfg.basis == "identity":
            self.U = torch.eye(dk, device=dev)[: self.K]
            return self.U
        covs = []
        for ks in self._keys:
            if ks:
                X = torch.stack(ks)
                covs.append(X.T @ X / len(ks))
            else:
                covs.append(torch.zeros(dk, dk, device=dev))
        C = torch.stack(covs)
        Cm = C.mean(0) if self.cfg.basis == "keys_svd" else coord_trimmed_mean(C, self.cfg.gate.f)
        _, evecs = torch.linalg.eigh(Cm)
        self.U = evecs[:, -self.K:].T.contiguous()   # frozen from now on
        return self.U

    def _grow_anchored(self) -> torch.Tensor:
        """Key-anchored, growing, frozen basis (v2.1).

        A key direction becomes a basis direction once >= k distinct sources
        wrote (a key within anchor_cos of) it during the last anchor_window
        rounds; so >= k - f honest sources must have written it, and keys
        written only by the adversary never become directions (Thm 1).  The
        new direction is the robust (coordinate-median) cluster centre,
        orthogonalised against the existing directions.  Existing directions
        never change, so the per-direction certificate stays valid; M^P lives
        in span(U) because it is only ever updated along U.
        """
        cfg = self.cfg
        self._hist.append([(s, k) for s, ks in enumerate(self._keys) for k in ks])
        self._hist = self._hist[-cfg.anchor_window:]
        need = cfg.gate.k if cfg.anchor_min_writers == "k" else cfg.gate.f + 1
        self.U = grow_anchored_basis(self.U, self._hist, cfg.n_sources, need, self.K,
                                     cfg.anchor_cos, cfg.anchor_min_resid)
        return self.U

    def promote(self) -> GateOutput | None:
        U = self.basis()                                   # (K, dk)
        n = self.cfg.n_sources
        self._keys = [[] for _ in range(n)]
        self.drift.resize(U.shape[0])
        if U.shape[0] == 0:
            self.Q.snapshot()
            self.rounds += 1
            return None
        Uq = U.unsqueeze(0).expand(n, *U.shape)
        cur = self.Q.read(Uq)                              # (n, K, dv)
        D = cur if self.cfg.rule == "ema" else cur - self.Q.read_snapshot(Uq)
        if self.delta_hook is not None:
            D = self.delta_hook(D, U)
        out = tabayyun_gate(D, self.cfg.gate)
        pm = out.promoted.to(self.MP.dtype).unsqueeze(1)   # (K,1)
        if self.cfg.rule == "ema":
            cur_P = U @ self.MP.T                          # (K, dv) = M^P u
            step = self.cfg.alpha_P * pm * (out.values - cur_P)
            self.MP = self.MP + step.T @ U
            if self.cfg.eps_P > 0:
                keep = (1 - pm) * (1 - self.cfg.eps_P) + pm   # decay unpromoted only
                self.MP = self.MP - ((1 - keep) * (U @ self.MP.T)).T @ U
        else:
            self.MP = (1 - self.cfg.alpha_P) * self.MP + out.values.T @ U
        self.drift.update(out.runtime_err, out.promoted)
        if self.oracle_honest is not None:
            self._update_reference(D, out, U, pm)
        if self.cfg.after_promotion == "reset":
            self.Q.reset()
        elif self.cfg.after_promotion == "decay":
            self.Q.decay(self.cfg.post_decay)
        self.Q.snapshot()
        self.rounds += 1
        self.last = out
        return out

    def _update_reference(self, D, out, U, pm):
        """M~^P: same decisions, honest-corroborator mean instead of TrMean (oracle)."""
        H = torch.zeros(D.shape[0], dtype=torch.bool, device=D.device)
        H[self.oracle_honest] = True
        CH = out.corroborators & H.unsqueeze(1)                     # (n,K)
        cnt = CH.sum(0).clamp(min=1).to(D.dtype).unsqueeze(1)
        ref_val = (D * CH.unsqueeze(-1)).sum(0) / cnt * pm          # (K,dv)
        if self.cfg.rule == "ema":
            cur = U @ self.MP_ref.T
            self.MP_ref = self.MP_ref + (self.cfg.alpha_P * pm * (ref_val - cur)).T @ U
            if self.cfg.eps_P > 0:
                keep = (1 - pm) * (1 - self.cfg.eps_P) + pm
                self.MP_ref = self.MP_ref - ((1 - keep) * (U @ self.MP_ref.T)).T @ U
        else:
            self.MP_ref = (1 - self.cfg.alpha_P) * self.MP_ref + ref_val.T @ U

    def true_drift(self) -> float:
        """||M^P - M~^P||_F (needs ``oracle_honest``)."""
        return float((self.MP - self.MP_ref).norm())

    # --------------------------------------------------------- certificate
    @property
    def certificate(self) -> float:
        """Frobenius drift certificate ||M^P - M~^P||_F <= this."""
        return self.drift.frobenius()

    def query_certificate(self, q: torch.Tensor) -> torch.Tensor:
        """Per-query bound on ||(M^P - M~^P) q||_2, q: (m, dk) -> (m,)."""
        return self.drift.query_bound(q, self.basis())
