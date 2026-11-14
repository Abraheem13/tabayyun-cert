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


@pytest.mark.parametrize("basis", ["identity", "anchored"])
@pytest.mark.parametrize("rule", ["ema", "additive"])
@pytest.mark.parametrize("attack", ["edge", "unbounded", "median_shift"])
def test_end_to_end_certificate_sound(rule, attack, basis):
    w = _world()
    cfg = TabayyunConfig(n_sources=10, dk=16, dv=16, tau=40, rule=rule, basis=basis,
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


def test_mean_field_law_reduces_to_lemma1_and_matches_day2():
    from tabayyun.certify import mean_field_half_life
    # lam = 0 -> Lemma 1 curve (values from the Day-2 exact computation)
    assert mean_field_half_life(0.0, 0.01, 0.0) == 69
    assert mean_field_half_life(0.9, 0.01, 0.0) == 79
    # Day-2 measured (linear, random stream, theta=0.05, d_k=64): 271.8 and 45.6
    lam = 2 * 0.05 / 64
    assert abs(mean_field_half_life(0.0, 0.001, lam) / 271.8 - 1) < 0.1
    assert abs(mean_field_half_life(0.9, 0.001, lam) / 45.6 - 1) < 0.1


def test_anchored_basis_recovers_fact_keys_and_ignores_adversary_keys():
    w = _world(n_facts=8, rho=0.9)
    mem = TabayyunMemory(TabayyunConfig(n_sources=10, dk=16, dv=16, tau=40, basis="anchored",
                                        gate=GateConfig(k=5, f=2, gamma=0.3, nu=0.05)))
    junk = torch.nn.functional.normalize(torch.randn(16), dim=0)
    atk = TargetedWrite(w.adv, junk, w.values[0])             # adversary-only key
    for _ in range(3):
        run_round(mem, w, atk)
    U = mem.basis()
    assert U.shape[0] >= 7                                     # fact keys found
    assert (U @ U.T - torch.eye(U.shape[0])).abs().max() < 1e-6  # orthonormal
    align = (U @ w.keys.T).abs().max(0).values               # each fact key ~ a direction
    assert (align > 0.97).float().mean() >= 7 / 8
    assert U.shape[0] == 8                                     # exactly the 8 fact keys
    assert float((U @ junk).abs().max()) < 0.9                 # adversary key not a direction



def test_ported_2026_baselines_run_and_shapes():
    from tabayyun.baselines import HuberSharedMemory, IsolateAggregateMemory, SMSRMemory
    w = _world()
    g = Gates(theta=0.5, alpha=0.005)
    mems = [HuberSharedMemory(16, 16, Gates(theta=0.5, eta=0.5, alpha=0.005)),
            SMSRMemory(10, 16, 16, g), IsolateAggregateMemory(10, 16, 16, g),
            TabayyunMemory(TabayyunConfig(n_sources=10, dk=16, dv=16, tau=40, q_gates=g, basis="anchored",
                                          gate=GateConfig(k=5, f=2, nu=0.05, cone=False, trim=False)))]
    for m in mems:
        run_round(m, w, TargetedWrite(w.adv, w.keys[0], -w.values[0]))
        y = m.read(w.keys[:3])
        assert y.shape == (3, 16) and torch.isfinite(y).all()


def test_untrimmed_gate_reports_no_certificate():
    D = torch.randn(7, 3, 4)
    out = tabayyun_gate(D, GateConfig(k=3, f=1, nu=0.0, cone=False, trim=False))
    assert (out.runtime_err[out.promoted] >= 1e8).all()


def test_memlm_scan_equals_titans_linear_memory():
    """The trained model's memory layer is exactly the certified Titans memory (constant gates)."""
    from tabayyun.models import MemLM
    g = torch.Generator().manual_seed(0)
    B, L, dk, dv = 2, 7, 8, 5
    k = torch.nn.functional.normalize(torch.randn(B, L, dk, generator=g), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, L, dk, generator=g), dim=-1)
    v = torch.randn(B, L, dv, generator=g)
    th, al, et = torch.full((B, L), 0.3), torch.full((B, L), 0.05), torch.full((B, L), 0.6)
    wm = torch.ones(B, L, dtype=torch.bool)
    y, M, _ = MemLM.scan(k, v, q, th, al, et, wm)
    ref = LinearMemory(B, dk, dv, Gates(theta=0.3, eta=0.6, alpha=0.05))
    for t in range(L):
        assert torch.allclose(y[:, t], ref.read(q[:, t]), atol=1e-10)
        ref.write(k[:, t], v[:, t])
    assert torch.allclose(M, ref.M, atol=1e-10)


def test_memlm_forward_and_readout_linear_in_y():
    from tabayyun.models import MemLM, MemLMConfig
    torch.manual_seed(0)
    m = MemLM(MemLMConfig(vocab=32, d_model=16, dk=8, dv=8)).double()
    h = torch.randn(3, 16, dtype=torch.float64)
    y1, y2 = torch.randn(3, 8, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64)
    W = m.readout_matrix()
    d = m.logits_from_read(h, y1) - m.logits_from_read(h, y2)
    assert torch.allclose(d, (y1 - y2) @ W.T, atol=1e-10)



def test_relative_activity_threshold_robust_to_adversary():
    """nu_rel: crosstalk-level sources become inactive; huge adversarial norms cannot
    push every honest supporter below the threshold."""
    g = torch.Generator().manual_seed(0)
    n, f, K, dv = 12, 2, 4, 6
    D = 0.02 * torch.randn(n, K, dv, generator=g)            # crosstalk for everybody
    base = torch.randn(K, dv, generator=g)
    sup = torch.arange(f, f + 6)                               # 6 honest supporters
    D[sup] = base + 0.05 * torch.randn(len(sup), K, dv, generator=g)
    D[:f] = 1000 * base                                        # adversary inflates norms
    out = tabayyun_gate(D, GateConfig(k=5, f=f, gamma=0.3, nu=1e-3, nu_rel=0.3))
    assert out.active[sup].all()                               # supporters stay active
    assert not out.active[f + 6:].any()                        # crosstalk-only sources dropped
    assert out.promoted.all()


def test_federated_composition_certificate_sound():
    """Theorem 5: global drift <= server runtime bound + max reported client bound,
    with Byzantine clients sending crafted rows AND under-reporting their bounds."""
    from tabayyun.attacks import EdgeOfRange as Edge
    from tabayyun.federated import FederatedTabayyun
    g = torch.Generator().manual_seed(0)
    N, n, f, f_c, dk, dv = 7, 6, 1, 1, 12, 12
    Q, _ = torch.linalg.qr(torch.randn(dk, dk, generator=g))
    keys = Q[:, :6].T.contiguous()
    vals = torch.nn.functional.normalize(torch.randn(6, dv, generator=g), dim=1)
    k_star, v_star = keys[0], -vals[0]
    clients = []
    for c in range(N):
        m = TabayyunMemory(TabayyunConfig(n_sources=n, dk=dk, dv=dv, tau=20, basis="anchored",
                                          anchor_min_writers="f+1", q_gates=Gates(theta=0.5, alpha=0.005),
                                          gate=GateConfig(k=3, f=f, gamma=0.3, nu=0.05)))
        m.oracle_honest = torch.arange(f, n)
        m.delta_hook = Edge([0], list(range(f, n)), k_star, v_star, nu=0.05)
        clients.append(m)
    fed = FederatedTabayyun(clients, GateConfig(k=3, f=f_c, gamma=0.3, nu=0.05))
    fed.byzantine = [0]
    fed.server_hook = Edge([0], list(range(1, N)), k_star, v_star, nu=0.05)
    for _ in range(4):
        for c, cl in enumerate(clients):
            for t in range(20):
                src, ks, vs = [], [], []
                for s in range(f, n):
                    j = int(torch.randint(0, 6, (1,), generator=g))
                    if torch.rand(1, generator=g) < 0.7:
                        src.append(s); ks.append(keys[j]); vs.append(vals[j] + 0.1 * torch.randn(dv, generator=g) / dv ** 0.5)
                if src:
                    cl.write(torch.tensor(src), torch.stack(ks), torch.stack(vs))
                else:
                    cl.write(torch.zeros(0, dtype=torch.long), torch.zeros(0, dk), torch.zeros(0, dv))
        fed.sync()
        assert fed.true_drift() <= fed.certificate + 1e-9
        dq = (keys @ (fed.MG - fed.MG_ref).T).norm(dim=-1)
        assert (dq <= fed.query_certificate(keys) + 1e-9).all()
    assert fed.U.shape[0] >= 3


def test_certified_regret_sound_fuzz():
    """Thm 3(a'): eps_hat bounds the reference regret of the chosen action."""
    from tabayyun.certify import certified_regret
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        A, dv = 20, 6
        y_ref = torch.randn(A, dv, generator=g)
        B = torch.rand(A, dv, generator=g) * 0.3
        y = y_ref + (2 * torch.rand(A, dv, generator=g) - 1) * B          # |y - y_ref| <= B
        w = torch.randn(dv, generator=g)
        u, u_ref = y @ w, y_ref @ w
        eps = certified_regret(u, B, w)
        a = int(u.argmax())
        assert float(u_ref.max() - u_ref[a]) <= eps + 1e-5


def test_action_certified_gram_never_less_strict():
    from tabayyun.certify import action_certified, action_certified_gram
    g = torch.Generator().manual_seed(1)
    n_cert = 0
    for _ in range(50):
        W = torch.randn(200, 24, generator=g)
        y = torch.randn(8, 24, generator=g) * 3
        logits = y @ W.T
        drift = torch.rand(8, generator=g) * 0.5
        a, b = action_certified(logits, W, drift), action_certified_gram(logits, W, drift)
        assert not (b & ~a).any()                 # gram-certified implies exactly certified
        n_cert += int(b.sum())
    assert n_cert > 0


def test_keymap_whitening_gives_near_orthogonal_keys():
    """Tier D key map: anisotropic features -> whitened unit keys with small cross-talk."""
    from tabayyun.models.lm_retrofit import KeyMap
    g = torch.Generator().manual_seed(0)
    A = torch.randn(96, 96, generator=g) * torch.linspace(5, 0.1, 96)        # anisotropic mixing
    feats = torch.randn(1500, 96, generator=g) @ A + 60.0                    # large common offset
    km = KeyMap(feats[:1000], 64)
    K = km(feats[1000:])
    assert torch.allclose(K.norm(dim=1), torch.ones(len(K)), atol=1e-5)
    raw = torch.nn.functional.normalize(feats[1000:], dim=1)
    off = lambda X: float((X[:200] @ X[:200].T - torch.eye(200)).abs().mean())  # noqa: E731
    assert off(raw) > 0.5 and off(K) < 0.15
