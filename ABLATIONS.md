# Ablation registry

Every ablation gets one row. Numbers come only from non-smoke runs in `results/registry.jsonl` (cite the `config_hash`).

| ID | Component removed / changed | Hypothesis tested | Experiment | config_hash | Result | Day |
|---|---|---|---|---|---|---|
| A1 | No quarantine (single tier) | Quarantine needed (Prop. 0 / H5) | E02 `titans*` | | | 4 |
| A2 | No corroboration gate (TrMean over all sources) | H6 | E02 `tab_nocorr` | Day 4 locality | **6/6 PASS** (worst ASR-or-amnesia +0.5 to +1.0 vs method) | 4 |
| A3 | Additive (v1) vs EMA (v2) promotion | F3: no 1/α^P | E02 `tabayyun_anchored2_add` | probe | additive Recall_L2 0.00 (fails); confirmatory Day 5 | 5 |
| A4 | Momentum in M^P | Thm 2 1/(1−η) penalty | E06 | | | 22 |
| A5 | ν, γ, β, k, τ sweeps | Operating curve | E06 | | | 22 |
| A6 | Basis: identity vs **anchored** (+ keys-SVD / robust-SVD later) | F5, F13 | E02 locality (`tabayyun` vs `tabayyun_anchored`) | Day 4 locality | **6/6 PASS** (recall −0.32 to −0.99 without anchoring) | 4 |
| A7 | Static vs learned trust | Proposal §9.6 | E06 | | | 23 |
| A8 | Linear vs random-feature promoted tier | Capacity vs certificate | E06 | | | 23 |
| A9 | Quarantine: linear vs MLP (deep quarantine) | F4 | E02/E03 | | | 10 |
| A10 | After-promotion policy: keep / reset / decay | Design | E06 | | | 22 |
| A11 | Baselines given the anchored basis too | Fairness of the locality result | E02 `agg_*_anc` | probe | still Recall_L2 0.00 (majority requirement); confirmatory Day 5 | 5 |
| A12 | Quarantine forget α^Q (clean-tuned) | Utility | E02 attack=none | | 0.005 selected (Recall_L2 1.0 for all) | 3 |
| A13 | Anchoring rule: k writers / 5-round window (v2.1) vs f+1 writers (v2.2) | Thm 3(b) bottleneck | E02 | Day 4 + probe | v2.2 matches Thm 3(b) within 0.07 (8/8) | 5 |
