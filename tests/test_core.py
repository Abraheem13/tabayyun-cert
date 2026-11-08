"""Unit tests: every theorem in the proposal is checked numerically.

These are *soundness* tests: a certificate that is violated even once is a
bug (or a wrong theorem), never noise.
"""
import math

import pytest
import torch

from tabayyun import (GateConfig, Gates, LinearMemory, MLPMemory, TabayyunConfig,
                      TabayyunMemory, action_certified, half_life, persistence_bound,
                      promotion_prob, tabayyun_gate)
from tabayyun.attacks import EdgeOfRange, MedianShift, TargetedWrite, Unbounded
from tabayyun.baselines import RobustAggMemory, SharedMemory
from tabayyun.data import FactWorld, WorldConfig, run_round
from tabayyun.metrics import benign_recall, targeted_success

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ memory
def test_delta_rule_one_shot():
    m = LinearMemory(3, 8, 5, Gates(theta=0.5))
    k = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    v = torch.randn(3, 5)
    m.write(k, v)
    assert torch.allclose(m.read(k), v, atol=1e-9)


def test_mask_freezes_other_sources():
    m = LinearMemory(4, 6, 6, Gates(theta=0.3, eta=0.5, alpha=0.1))
    k, v = torch.randn(4, 6), torch.randn(4, 6)
    mask = torch.tensor([True, False, True, False])
    m.write(k, v, mask=mask)
    assert m.M[1].abs().sum() == 0 and m.M[3].abs().sum() == 0
    assert m.M[0].abs().sum() > 0


@pytest.mark.parametrize("eta,alpha", [(0.0, 0.1), (0.5, 0.05), (0.9, 0.02), (0.95, 0.2)])
def test_persistence_lemma_exact(eta, alpha):
    """Lemma 1: one write's contribution evolves as theta g sum (1-a)^{T-j} eta^j,
    when later writes are orthogonal to it."""
    dk = 16
    m = LinearMemory(1, dk, 4, Gates(theta=0.1, eta=eta, alpha=alpha))
    k0 = torch.zeros(1, dk); k0[0, 0] = 1
    v0 = torch.ones(1, 4)
    m.write(k0, v0)
    g0 = float((0.1 * 2 * v0).norm())                  # ||theta g|| at t=0 (M=0)
    for T in range(1, 40):
        k = torch.zeros(1, dk); k[0, 1 + (T % (dk - 1))] = 1
        m.write(k, torch.randn(1, 4))
        got = float(m.read(k0).norm())
        assert math.isclose(got, persistence_bound(g0, eta, alpha, T), rel_tol=1e-8, abs_tol=1e-12)


def test_half_life_monotone_in_momentum():
    assert half_life(0.9, 0.01) > half_life(0.0, 0.01) or half_life(0.0, 0.01) == half_life(0.9, 0.01)
    assert half_life(0.95, 0.5) > half_life(0.5, 0.5)


def test_mlp_memory_learns():
    m = MLPMemory(2, 8, 4, Gates(theta=0.2), hidden=32)
    k = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    v = torch.randn(2, 4)
    e0 = float((m.read(k) - v).norm())
    for _ in range(200):
        m.write(k, v)
    assert float((m.read(k) - v).norm()) < 0.1 * e0


# -------------------------------------------------------------------- gate
def _honest_block(n_h, K, dv, spread, active_dirs, g):
    base = torch.randn(K, dv, generator=g)
    D = base.unsqueeze(0) + spread * (torch.rand(n_h, K, dv, generator=g) - 0.5)
    D[:, ~active_dirs] = 0
    return D


def test_gate_requires_valid_k():
    with pytest.raises(ValueError):
        GateConfig(k=2, f=1)


def test_theorem1_no_honest_support_zero_influence():
    g = torch.Generator().manual_seed(0)
    n_h, f, K, dv = 7, 2, 6, 5
    active = torch.tensor([True, True, True, False, False, False])
    H = _honest_block(n_h, K, dv, 0.1, active, g)
    A = 50 * torch.randn(f, K, dv, generator=g)          # arbitrary adversarial rows
    out = tabayyun_gate(torch.cat([A, H]), GateConfig(k=5, f=f, gamma=0.3))
    assert not out.promoted[~active].any()
    assert (out.values[~active] == 0).all()


@pytest.mark.parametrize("seed", range(30))
def test_hull_and_runtime_certificate_fuzz(seed):
    """Promoted values stay in the honest-corroborator hull; runtime error
    bound e holds against arbitrary adversarial rows."""
    g = torch.Generator().manual_seed(seed)
    n_h, f, K, dv = 8, 3, 5, 4
    H = _honest_block(n_h, K, dv, 0.3, torch.ones(K, dtype=torch.bool), g)
    mode = seed % 3
    if mode == 0:
        A = 100 * torch.randn(f, K, dv, generator=g)
    elif mode == 1:   # inside-the-cone edge attack
        A = H.max(0).values.unsqueeze(0).repeat(f, 1, 1) + 0.0
    else:
        A = H.mean(0, keepdim=True).repeat(f, 1, 1) * 1.5
    D = torch.cat([A, H])
    cfg = GateConfig(k=7, f=f, gamma=0.2)
    out = tabayyun_gate(D, cfg)
    Hmask = torch.zeros(D.shape[0], dtype=torch.bool); Hmask[f:] = True
    for u in torch.nonzero(out.promoted).flatten().tolist():
        CH = out.corroborators[:, u] & Hmask
        hv = D[CH, u]
        assert (out.values[u] >= hv.min(0).values - 1e-9).all()
        assert (out.values[u] <= hv.max(0).values + 1e-9).all()
        mu = hv.mean(0)
        assert ((out.values[u] - mu).abs() <= out.runtime_err[u] + 1e-9).all()


def test_promotion_prob_matches_simulation():
    g = torch.Generator().manual_seed(1)
    n_H, rho, r, k = 8, 0.3, 3, 4
    sims = (torch.rand(20000, n_H, r, generator=g) < rho).any(-1).sum(-1) >= k
    assert abs(float(sims.double().mean()) - promotion_prob(n_H, rho, r, k)) < 0.015


# ------------------------------------------------------- end-to-end memory
def _world(**kw):
    base = dict(n_sources=10, n_adv=2, dk=16, dv=16, n_facts=12, rho=0.7, sigma=0.1, tau=40, seed=0)
    base.update(kw)
    return FactWorld(WorldConfig(**base))


@pytest.mark.parametrize("rule", ["ema", "additive"])
@pytest.mark.parametrize("attack", ["edge", "unbounded", "median_shift"])
def test_end_to_end_certificate_sound(rule, attack):
    w = _world()
    cfg = TabayyunConfig(n_sources=10, dk=16, dv=16, tau=40, rule=rule,
                         alpha_P=0.5 if rule == "ema" else 0.1,
                         gate=GateConfig(k=5, f=2, gamma=0.3, nu=0.05))
    mem = TabayyunMemory(cfg)
    mem.oracle_honest = w.honest
    k_star, v_star = w.keys[0], -w.values[0]
    hook = {"edge": EdgeOfRange(w.adv, w.honest, k_star, v_star, nu=0.05),
            "unbounded": Unbounded(w.adv, k_star, v_star),
            "median_shift": MedianShift(w.adv)}[attack]
    mem.delta_hook = hook
    for _ in range(8):
        run_round(mem, w)
        assert mem.true_drift() <= mem.certificate + 1e-9
        q = w.keys
        dq = (q @ (mem.MP - mem.MP_ref).T).norm(dim=-1)
        assert (dq <= mem.query_certificate(q) + 1e-9).all()


def test_tabayyun_blocks_minority_insertion_where_shared_memory_fails():
    w = _world(n_facts=8)
    k_star = w.keys[0]
    v_star = -w.values[0]
    gates = Gates(theta=0.5, alpha=0.02)
    tab = TabayyunMemory(TabayyunConfig(n_sources=10, dk=16, dv=16, tau=40, q_gates=gates,
                                        gate=GateConfig(k=5, f=2, gamma=0.3, nu=0.05)))
    shared = SharedMemory(16, 16, gates)
    atk = TargetedWrite(w.adv, k_star, v_star)
    for _ in range(6):
        run_round(tab, w, atk)
        run_round(shared, w, atk)
    assert targeted_success(shared, k_star, v_star, w.values[0])
    assert not targeted_success(tab, k_star, v_star, w.values[0])
    assert benign_recall(tab, w.keys[1:], w.values[1:]) >= 0.85


def test_action_certificate_sound():
    g = torch.Generator().manual_seed(3)
    A, dv, B = 5, 8, 200
    W = torch.randn(A, dv, generator=g)
    y = torch.randn(B, dv, generator=g)
    D = 0.2 * torch.ones(B)
    cert = action_certified(y @ W.T, W, D)
    a0 = (y @ W.T).argmax(-1)
    for _ in range(50):
        d = torch.randn(B, dv, generator=g)
        d = d / d.norm(dim=1, keepdim=True) * D.unsqueeze(1) * torch.rand(B, 1, generator=g)
        a1 = ((y + d) @ W.T).argmax(-1)
        assert (a1[cert] == a0[cert]).all()
    assert cert.any()


def test_robust_agg_baseline_runs():
    w = _world()
    for agg in ["mean", "trmean", "median", "krum", "geomed", "cclip"]:
        m = RobustAggMemory(10, 16, 16, Gates(theta=0.5, alpha=0.02), agg=agg, f=2, tau=40)
        run_round(m, w, TargetedWrite(w.adv, w.keys[0], -w.values[0]))
        assert torch.isfinite(m.MP).all()
