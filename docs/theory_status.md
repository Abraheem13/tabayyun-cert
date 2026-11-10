# Theory status

A result may be called a *Theorem* in the paper only when its proof is written in `paper/proofs.tex` **and** numerically checked by a test.

| Result | Statement (short) | Proof | Numerical check | Status |
|---|---|---|---|---|
| Lemma 1 | Single-write persistence, exact for orthogonal streams | Sketch (proposal) | `test_persistence_lemma_exact` (rel. err ≤ 1e-8) | ✅ checked |
| Lemma 1′ | Mean-field persistence with benign self-correction λ = 2θ/d_k; momentum amplifies correction | Derivation to write (expected dynamics of a 2×2 linear system) | Day-2 E01: 42/42 linear cells within −9%…+4%, log-log r = 0.9996; `test_mean_field_law_reduces_to_lemma1_and_matches_day2` | ✅ empirical, proof Day 3 |
| Prop. 0 | Necessity: a single-tier memory cannot bound one source under any norm filter | To write (rayn construction) | E01 rayn (smoke: cos 0.99 at 10% norm) | ⏳ proof Day 3 |
| Thm 1 | Sybil-limited promotion, zero influence on unsupported directions | Sketch | `test_theorem1_no_honest_support_zero_influence` | ✅ checked |
| Lemma 2 | Hull of honest corroborators | Standard order statistics | `test_hull_and_runtime_certificate_fuzz` (30 seeds) | ✅ checked |
| Thm 2′ | EMA drift ≤ √(d_v K) Δ_H^C, with no 1/α^P | To write | via Prop. R test | ⏳ Day 5 |
| Prop. R | Runtime certificate, per query | Order-statistic argument (in `gate.py` docstring) | `test_end_to_end_certificate_sound` (2 rules × 3 attacks × 8 rounds) | ✅ checked |
| Thm 3a | Pairwise action certificate | Short | `test_action_certificate_sound` | ✅ checked |
| Thm 3b | Promotion-delay law | Short | `test_promotion_prob_matches_simulation` | ✅ checked |
| Thm 4 | Tightness, Ω(f/(c−2β)·Δ_H^C) | Open | E02 tightness ratio | 🔲 Days 26–28 |
| Prop. 5 | Liveness under median shift | Open | E02 promotion fraction under `MedianShift` | 🔲 Day 20 |
| Fed. composition | Two-level certificate | Open | E05 | 🔲 Day 16 |
