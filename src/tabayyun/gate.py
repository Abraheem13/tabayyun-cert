"""The tabayyun gate: corroboration-gated, trimmed promotion (Sec. 6.3, v2).

Input: per-source read-deltas along each basis direction
    D[s, u, :] = M_s^Q(u) - M_s^{Q,prev}(u)          shape (n, K, dv)
(for a linear memory this is exactly Delta_s u).

For each direction u:
    A_u   = { s : ||D[s,u]|| >= nu }                      active sources
    m_u   = coord-median_{s in A_u} D[s,u]                (v2: over A_u only)
    C_u   = { s in A_u : cos(D[s,u], m_u) >= gamma }      corroborators
    promote u iff |C_u| >= k   (k >= max(2*beta+1, f+1))
    value_u = TrMean_beta over C_u                        (v2: over C_u only)

v1 of the proposal took the median and the trimmed mean over *all* n
sources.  Honest sources that never observed the fact contribute zeros, so
(i) the median is ~0 and the cosine test is undefined, and (ii) the honest
spread Delta_H becomes ~||v||, which makes Theorem 2 vacuous.  v2 fixes both.

Also returns the *runtime-computable* certificate (Proposition R in
docs/proposal_v2.md): with at most f adversarial values among the c = |C_u|
corroborators, the honest-corroborator mean of every coordinate lies in
[L, U] where L (U) is the mean of the c - f smallest (largest) values, so

    |value_u,i - mu_H,u,i| <= e_u,i := max(value_u,i - L_i, U_i - value_u,i).

Unlike the a-priori bound, e needs no knowledge of Delta_H or of which
sources are honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GateConfig:
    k: int = 3            # corroboration threshold
    f: int = 1            # assumed max number of adversarial sources
    beta: int | None = None  # trimming per side (default: f)
    gamma: float = 0.5    # cosine agreement with the robust median
    nu: float = 1e-3      # activity threshold on ||D[s,u]||
    corroborate: bool = True  # ablation: False -> trimmed mean only, no gate

    def __post_init__(self):
        if self.beta is None:
            self.beta = self.f
        if self.corroborate and self.k < max(2 * self.beta + 1, self.f + 1):
            raise ValueError(
                f"need k >= max(2*beta+1, f+1); got k={self.k}, beta={self.beta}, f={self.f}")


def _masked_sorted(D: torch.Tensor, member: torch.Tensor):
    """Sort values of members along the source axis; non-members -> +inf (last).

    D: (n, K, dv), member: (n, K) bool.  Returns sorted (n, K, dv), counts (K,).
    """
    big = torch.where(member.unsqueeze(-1), D, torch.full_like(D, float("inf")))
    srt, _ = torch.sort(big, dim=0)
    return srt, member.sum(0)


def _range_mean(srt: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Mean over sorted positions [lo, hi) per direction. lo, hi: (K,)."""
    n = srt.shape[0]
    pos = torch.arange(n, device=srt.device).view(n, 1)
    w = ((pos >= lo.view(1, -1)) & (pos < hi.view(1, -1))).to(srt.dtype)  # (n,K)
    cnt = w.sum(0).clamp(min=1)
    vals = torch.where(torch.isfinite(srt), srt, torch.zeros_like(srt))
    return (vals * w.unsqueeze(-1)).sum(0) / cnt.unsqueeze(-1)


def masked_median(D: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise lower median over members. Returns (K, dv)."""
    srt, cnt = _masked_sorted(D, member)
    idx = ((cnt - 1).clamp(min=0) // 2).view(1, -1, 1).expand(1, *srt.shape[1:])
    med = torch.gather(srt, 0, idx).squeeze(0)
    return torch.where(torch.isfinite(med), med, torch.zeros_like(med))


@dataclass
class GateOutput:
    values: torch.Tensor      # (K, dv) promoted value per direction (0 if not promoted)
    promoted: torch.Tensor    # (K,) bool
    active: torch.Tensor      # (n, K) bool
    corroborators: torch.Tensor  # (n, K) bool
    counts: torch.Tensor      # (K,) |C_u|
    runtime_err: torch.Tensor  # (K, dv) e_{u,i} (0 where not promoted)


def tabayyun_gate(D: torch.Tensor, cfg: GateConfig) -> GateOutput:
    n, K, dv = D.shape
    norms = D.norm(dim=-1)                                  # (n,K)
    active = norms >= cfg.nu
    if cfg.corroborate:
        med = masked_median(D, active)                      # (K,dv)
        cos = torch.nn.functional.cosine_similarity(D, med.unsqueeze(0), dim=-1, eps=1e-12)
        medn = med.norm(dim=-1)
        C = active & (cos >= cfg.gamma) & (medn > 0).unsqueeze(0)
        promoted = C.sum(0) >= cfg.k
    else:
        # ablation "trimmed mean only": every source counts, every direction promoted
        C = torch.ones_like(active)
        promoted = torch.ones(K, dtype=torch.bool, device=D.device)
    srt, c = _masked_sorted(D, C)
    b = cfg.beta
    values = _range_mean(srt, torch.full_like(c, b), c - b)
    hf = (c - cfg.f).clamp(min=1)
    L = _range_mean(srt, torch.zeros_like(c), hf)
    U = _range_mean(srt, c - hf, c)
    err = torch.maximum(values - L, U - values).clamp(min=0)
    pm = promoted.view(K, 1).to(D.dtype)
    return GateOutput(values * pm, promoted, active, C, c, err * pm)


# ---------------------------------------------------------------- baselines
def coord_trimmed_mean(X: torch.Tensor, beta: int) -> torch.Tensor:
    """Yin et al. 2018. X: (n, ...)."""
    n = X.shape[0]
    srt, _ = torch.sort(X, dim=0)
    return srt[beta:n - beta].mean(0)


def coord_median(X: torch.Tensor) -> torch.Tensor:
    return X.median(dim=0).values


def krum(X: torch.Tensor, f: int, m: int = 1) -> torch.Tensor:
    """(Multi-)Krum, Blanchard et al. 2017. X: (n, ...)."""
    n = X.shape[0]
    flat = X.flatten(1)
    d = torch.cdist(flat, flat) ** 2
    nb = max(n - f - 2, 1)
    scores = torch.sort(d, dim=1).values[:, 1:nb + 1].sum(1)
    idx = torch.argsort(scores)[:m]
    return X[idx].mean(0)


def geometric_median(X: torch.Tensor, iters: int = 50, eps: float = 1e-8) -> torch.Tensor:
    """Weiszfeld; RFA (Pillutla et al. 2022)."""
    flat = X.flatten(1)
    z = flat.mean(0)
    for _ in range(iters):
        w = 1.0 / (flat - z).norm(dim=1).clamp(min=eps)
        z = (w.unsqueeze(1) * flat).sum(0) / w.sum()
    return z.view(X.shape[1:])


def centered_clipping(X: torch.Tensor, ref: torch.Tensor, tau: float, iters: int = 3) -> torch.Tensor:
    """Karimireddy et al. 2021."""
    v = ref.clone()
    for _ in range(iters):
        diff = X - v
        nrm = diff.flatten(1).norm(dim=1).clamp(min=1e-12)
        scale = torch.clamp(tau / nrm, max=1.0).view(-1, *([1] * (X.dim() - 1)))
        v = v + (diff * scale).mean(0)
    return v


def fltrust(X: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """Cao et al. 2021: ReLU-cosine trust w.r.t. a server root update."""
    flat, r = X.flatten(1), root.flatten()
    ts = torch.relu(torch.nn.functional.cosine_similarity(flat, r.unsqueeze(0), dim=1))
    scaled = flat * (r.norm() / flat.norm(dim=1).clamp(min=1e-12)).unsqueeze(1)
    out = (ts.unsqueeze(1) * scaled).sum(0) / ts.sum().clamp(min=1e-12)
    return out.view(X.shape[1:])


AGGREGATORS = {
    "mean": lambda X, f, **kw: X.mean(0),
    "trmean": lambda X, f, **kw: coord_trimmed_mean(X, f),
    "median": lambda X, f, **kw: coord_median(X),
    "krum": lambda X, f, **kw: krum(X, f, m=1),
    "multikrum": lambda X, f, **kw: krum(X, f, m=max(X.shape[0] - f, 1)),
    "geomed": lambda X, f, **kw: geometric_median(X),
}
