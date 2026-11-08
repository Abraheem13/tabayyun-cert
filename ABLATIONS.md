# Ablation registry

Every ablation gets one row. Numbers come only from non-smoke runs in `results/registry.jsonl` (cite the `config_hash`).

| ID | Component removed / changed | Hypothesis tested | Experiment | config_hash | Result | Day |
|---|---|---|---|---|---|---|
| A1 | No quarantine (single tier) | Quarantine needed (Prop. 0 / H5) | E02 `titans*` | | | 4 |
| A2 | No corroboration gate (TrMean over all sources) | H6 | E02 `agg_trmean` + `corroborate=False` | | | 4 |
| A3 | Additive (v1) vs EMA (v2) promotion | F3: no 1/α^P | E02 `--rule` | | | 5 |
| A4 | Momentum in M^P | Thm 2 1/(1−η) penalty | E06 | | | 22 |
| A5 | ν, γ, β, k, τ sweeps | Operating curve | E06 | | | 22 |
| A6 | Basis: identity / keys-SVD / robust-SVD | F5 | E06 | | | 23 |
| A7 | Static vs learned trust | Proposal §9.6 | E06 | | | 23 |
| A8 | Linear vs random-feature promoted tier | Capacity vs certificate | E06 | | | 23 |
| A9 | Quarantine: linear vs MLP (deep quarantine) | F4 | E02/E03 | | | 10 |
| A10 | After-promotion policy: keep / reset / decay | Design | E06 | | | 22 |
