"""Certificates (Sec. 7 of docs/proposal_v2.md).

* ``persistence_bound``   Lemma 1: norm of one write's contribution after T steps.
* ``half_life``           ln 2 / (-ln max(eta, 1 - alpha)).
* ``apriori_drift``       Theorem 2 (v2): D_R from the honest-corroborator spread.
* ``RuntimeDrift``        Proposition R (scalar form, v1 additive rule).
* ``DirectionalDrift``    Theorem 2' (v2): per-direction, per-coordinate
                          runtime certificate and per-query bound.
* ``action_certified``    Theorem 3(a), pairwise (tighter) form for a linear head.
* ``promotion_prob``      Theorem 3(b): benign-promotion delay law.
"""
from __future__ import annotations

import math

import torch
from scipy.stats import binom


def persistence_bound(theta_G: float, eta: float, alpha: float, T: int) -> float:
    """Exact norm of sum_{j=0}^T (1-alpha)^{T-j} eta^j times theta*G."""
    a = 1.0 - alpha
    s = sum(a ** (T - j) * eta ** j for j in range(T + 1))
    return theta_G * s


def persistence_upper(theta_G: float, eta: float, alpha: float, T: int) -> float:
    c = max(eta, 1.0 - alpha)
    b1 = theta_G * (T + 1) * c ** T
    b2 = theta_G / (alpha * (1 - eta)) if alpha > 0 and eta < 1 else math.inf
    return min(b1, b2)


def half_life(eta: float, alpha: float) -> float:
    c = max(eta, 1.0 - alpha)
    if c <= 0:
        return 0.0
    if c >= 1:
        return math.inf
    return math.log(2) / (-math.log(c))


def mean_field_half_life(eta: float, alpha: float, lam: float, T: int = 200000) -> float:
    """Lemma 1' (mean-field persistence law, Day 2).

    Along k*, a poisoned write's footprint x_t in a stream of benign writes with
    random unit keys follows, in expectation,
        S_t = eta S_{t-1} - lam x_{t-1},     x_t = (1 - alpha) x_{t-1} + S_t,
    with lam = 2 theta E[<k, k*>^2] = 2 theta / d_k (self-correction by benign
    writes, which momentum *amplifies*).  lam = 0 recovers Lemma 1 (orthogonal
    stream).  Returns steps from the peak of |x_t| to half of it.
    """
    S, x = 1.0, 1.0
    peak, pk = 1.0, 0
    for t in range(1, T + 1):
        S = eta * S - lam * x
        x = (1 - alpha) * x + S
        if abs(x) > peak:
            peak, pk = abs(x), t
        elif abs(x) <= 0.5 * peak:
            return float(t - pk)
    return math.inf


def apriori_drift(dv: int, K: int, delta_H: float, alpha_P: float, R: int | None = None) -> float:
    """Theorem 2 (v2). delta_H = honest-corroborator coordinate spread."""
    per_round = math.sqrt(dv * K) * delta_H
    if alpha_P <= 0:
        return per_round * (R if R is not None else math.inf)
    if R is None:
        return per_round / alpha_P
    return per_round * (1 - (1 - alpha_P) ** R) / alpha_P


class RuntimeDrift:
    """E_R = (1 - alpha_P) E_{R-1} + ||e_R||_F   (Proposition R).

    Because promoted directions are orthonormal, the Frobenius error of one
    round is the l2 norm of the per-direction, per-coordinate errors.
    """

    def __init__(self, alpha_P: float):
        self.alpha_P = alpha_P
        self.E = 0.0
        self.history: list[float] = []

    def update(self, runtime_err: torch.Tensor) -> float:
        e = float(runtime_err.norm())
        self.E = (1 - self.alpha_P) * self.E + e
        self.history.append(self.E)
        return self.E


class DirectionalDrift:
    """Per-direction, per-coordinate drift certificate (Theorem 2', v2).

    Tracks E in R^{K x dv} with |(M^P u - M~^P u)_i| <= E[u, i], where M~^P is
    the promoted memory that would result from using the honest-corroborator
    mean instead of the trimmed mean, under the same promotion decisions.

      ema      : promoted   E <- (1-alpha) E + alpha e ;  unpromoted E <- (1-eps) E
      additive : promoted   E <- (1-alpha) E + e       ;  unpromoted E <- (1-alpha) E
    """

    def __init__(self, K: int, dv: int, rule: str, alpha_P: float, eps_P: float = 0.0,
                 device="cpu"):
        self.E = torch.zeros(K, dv, device=device)
        self.rule, self.a, self.eps = rule, alpha_P, eps_P
        self.history: list[float] = []

    def resize(self, K: int) -> None:
        """Grow for newly added (frozen) directions; their error starts at 0
        because M^P and the reference are both 0 along a new direction."""
        if K > self.E.shape[0]:
            pad = torch.zeros(K - self.E.shape[0], self.E.shape[1], device=self.E.device, dtype=self.E.dtype)
            self.E = torch.cat([self.E, pad])

    def update(self, e: torch.Tensor, promoted: torch.Tensor) -> None:
        pm = promoted.to(self.E.dtype).unsqueeze(1)
        if self.rule == "ema":
            self.E = pm * ((1 - self.a) * self.E + self.a * e) + (1 - pm) * (1 - self.eps) * self.E
        else:
            self.E = (1 - self.a) * self.E + pm * e
        self.history.append(self.frobenius())

    def frobenius(self) -> float:
        return float(self.E.norm())

    def query_bound(self, q: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """||(M^P - M~^P) q||_2 <= || sum_u |<u,q>| E_u ||_2  (tighter than ||E||_F ||q||)."""
        w = (q @ U.T).abs()                     # (m, K)
        return (w @ self.E).norm(dim=-1)

    def query_coord_bound(self, q: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Per-coordinate: |(M^P - M~^P) q|_i <= sum_u |<u,q>| E_{u,i}.  Returns (m, dv)."""
        return (q @ U.T).abs() @ self.E


def action_certified(logits: torch.Tensor, W: torch.Tensor, drift_q: torch.Tensor) -> torch.Tensor:
    """Theorem 3(a), pairwise form for a linear head z = W y + b.

    logits: (B, A); W: (A, dv); drift_q: (B,) upper bound on ||y - y~|| for the
    query (e.g. ``TabayyunMemory.query_certificate`` or ||E||_F * ||q||).
    The action a* = argmax z is certified iff, for all j != a*,
        z_a* - z_j > ||w_a* - w_j|| * drift_q .
    Returns (B,) bool.
    """
    top = logits.argmax(-1)
    za = logits.gather(1, top.view(-1, 1))
    wdiff = (W[top].unsqueeze(1) - W.unsqueeze(0)).norm(dim=-1)   # (B, A)
    ok = (za - logits) > wdiff * drift_q.view(-1, 1)
    ok.scatter_(1, top.view(-1, 1), True)
    return ok.all(-1)


def action_certified_gram(logits: torch.Tensor, W: torch.Tensor, drift_q: torch.Tensor) -> torch.Tensor:
    """Theorem 3(a) for large vocabularies: same test as ``action_certified`` but with
    ||w_a - w_j|| from the Gram identity (O(A d) memory instead of O(B A d)).
    The distance is rounded UP by a relative 1e-4, so this is never less strict.
    """
    top = logits.argmax(-1)
    n2 = (W * W).sum(-1)                                             # (A,)
    d2 = n2[top].unsqueeze(1) + n2.unsqueeze(0) - 2.0 * (W[top] @ W.T)   # (B, A)
    wdiff = d2.clamp(min=0).sqrt() * (1 + 1e-4) + 1e-4 * n2.max().sqrt()
    za = logits.gather(1, top.view(-1, 1))
    ok = (za - logits) > wdiff * drift_q.view(-1, 1)
    ok.scatter_(1, top.view(-1, 1), True)
    return ok.all(-1)


def certified_regret(util: torch.Tensor, coord_bound: torch.Tensor, w: torch.Tensor) -> float:
    """Theorem 3(a'), certified epsilon-optimality for a linear utility u_a = w . y_a.

    util: (A,) utilities of the reads; coord_bound: (A, dv) per-coordinate bounds on
    |y_a - y~_a|; w: (dv,).  With s_a = sum_i |w_i| B_{a,i} >= |u_a - u~_a|,
        max_j u~_j - u~_{a_hat} <= eps_hat = max_j (u_j - u_{a_hat} + s_j + s_{a_hat})^+,
    a_hat = argmax_a u_a.  eps_hat = 0 iff a_hat is certified exactly (Thm 3a).
    """
    s = coord_bound @ w.abs()
    a = int(util.argmax())
    return float((util - util[a] + s + s[a]).clamp(min=0).max())


def action_certified_lipschitz(margin: torch.Tensor, L: float, q_norm: torch.Tensor,
                               D: float) -> torch.Tensor:
    """Theorem 3(a) as in v1: m(q) > 2 L ||q|| D (for nonlinear heads)."""
    return margin > 2 * L * q_norm * D


def promotion_prob(n_H: int, rho: float, r: int, k: int) -> float:
    """Theorem 3(b): P(Bin(n_H, 1-(1-rho)^r) >= k)."""
    p = 1 - (1 - rho) ** r
    return float(binom.sf(k - 1, n_H, p))
