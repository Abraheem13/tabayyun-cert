# Ablation registry

Every ablation gets one row. Numbers come only from non-smoke runs in `results/registry.jsonl` (cite the `config_hash`).

| ID | Component removed / changed | Hypothesis tested | Experiment | config_hash | Result | Day |
|---|---|---|---|---|---|---|
| A1 | No quarantine (single tier) | Quarantine needed (Prop. 0 / H5) | E02 `titans*` | | | 4 |
| A2 | No corroboration gate (TrMean over all sources) | H6 | E02 `tab_nocorr` | Day 4 locality | **6/6 PASS** (worst ASR-or-amnesia +0.5 to +1.0 vs method) | 4 |
| A3 | Additive (v1) vs EMA (v2) promotion | F3: no 1/α^P | E02 `tabayyun_anchored2_add` | probe | additive Recall_L2 0.00 (fails); confirmatory Day 5 | 5 |
| A4 | Momentum in M^P (η_P ∈ {0.5, 0.9}) | Thm 2′ needs a momentum-free promoted tier | E07 `tab_etaP*` | e07_amend7 (seeds 500–509) | **As predicted: the certificate breaks.** 40/200 runs violate at each η_P; drift/cert up to 1.45 (η_P = 0.5) and 2.13 (η_P = 0.9). η_P = 0.9 also costs recall (locality 0.99 → 0.41) | 12 |
| A5 | ν ∈ {0.02, 0.1, 0.2}, γ ∈ {0.1, 0.6}, no cone, k + 2, τ ∈ {25, 100}, α_P ∈ {0.2, 0.9} | Operating curve | E07 | e07_amend7 | 0 violations in every variant. **ν, γ, τ, α_P are inert** in Tier A (recall within 0.02, installed 0). **No cone:** still sound and installed 0 (my prediction of higher installed was **wrong**), but the **certificate is 20× looser** (Frobenius 4.62 vs 0.23); the cone test buys tightness. **k + 2:** locality recall 0.99 → 0.58, worst flip 0.10, as Thm 3(b) predicts | 12 |
| A6 | Basis: identity vs **anchored** (+ keys-SVD / robust-SVD later) | F5, F13 | E02 locality (`tabayyun` vs `tabayyun_anchored`) | Day 4 locality | **6/6 PASS** (recall −0.32 to −0.99 without anchoring) | 4 |
| A7 | Static vs learned trust | Proposal §9.6 | E06 | | | 23 |
| A8 | Linear vs random-feature promoted tier | Capacity vs certificate | E06 | | | 23 |
| A9 | Quarantine: linear vs MLP (deep quarantine) | F4 | E02/E03 | | | 10 |
| A10 | After-promotion policy: keep / reset / decay | Design | E07 `tab_reset`, `tab_decay` | e07_amend7 | **keep is best.** reset: locality recall 0.61, amnesia 0.90. decay: locality 0.82, dense 0.95, certificate 3× looser. All sound | 12 |
| A11 | Baselines given the anchored basis too | Fairness of the locality result | E02 `agg_*_anc` | probe | still Recall_L2 0.00 (majority requirement); confirmatory Day 5 | 5 |
| A12 | Quarantine forget α^Q (clean-tuned) | Utility | E02 attack=none | | 0.005 selected (Recall_L2 1.0 for all) | 3 |
| A13 | Anchoring rule: k writers / 5-round window (v2.1) vs f+1 writers (v2.2) | Thm 3(b) bottleneck | E02 | Day 4 + probe | v2.2 matches Thm 3(b) within 0.07 (8/8) | 5 |
