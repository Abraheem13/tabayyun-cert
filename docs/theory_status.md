# Theory status

A result may be called a *Theorem* in the paper only when its proof is written in `paper/proofs.tex` **and** numerically checked by a test.

| Result | Statement (short) | Proof | Numerical check | Status |
|---|---|---|---|---|
| Lemma 1 | Single-write persistence, exact for orthogonal streams | `paper/proofs.md` | unit test (1e-8) + Day 3 E01 orthogonal: 42/42 cells exact | ✅ proved + checked |
| Lemma 1′ | Mean-field persistence with benign self-correction λ = 2θ/d_k; momentum amplifies correction; MLP α_eff ≈ 2α | `paper/proofs.md` | Day 2: 42/42 linear cells −9%…+4%, r = 0.9996; Day 3 MLP r = 0.994 (exploratory) | ✅ proof written |
| Prop. 0 | Necessity: norm filters cannot bound one source; contest becomes write frequency | Sketch in `paper/proofs.md` | Day 3: rayn ASR 1.0 under 6 filters (20/20 each), single-shot 0 | ✅ sketch + checked; full proof Day 7 |
| Thm 1 | Sybil-limited promotion, zero influence on unsupported directions | Sketch | `test_theorem1_no_honest_support_zero_influence` | ✅ checked |
| Lemma 2 | Hull of honest corroborators | Standard order statistics | `test_hull_and_runtime_certificate_fuzz` (30 seeds) | ✅ checked |
| Thm 2′ | EMA drift ≤ √(d_v K) Δ_H^C, with no 1/α^P | To write | via Prop. R test | ⏳ Day 5 |
| Prop. R | Runtime certificate, per query | Order-statistic argument (in `gate.py` docstring) | `test_end_to_end_certificate_sound` (2 rules × 3 attacks × 8 rounds) | ✅ checked |
| Thm 3a | Pairwise action certificate | Short | `test_action_certificate_sound` | ✅ checked |
| Thm 3b | Promotion-delay law | Short | `test_promotion_prob_matches_simulation` | ✅ checked |
| Thm 4 | Tightness, Ω(f/(c−2β)·Δ_H^C) | Open | E02 tightness ratio | 🔲 Days 26–28 |
| Prop. 5 | Liveness under median shift | Open | E02 promotion fraction under `MedianShift` | 🔲 Day 20 |
| Fed. composition | Two-level certificate | Open | E05 | 🔲 Day 16 |
