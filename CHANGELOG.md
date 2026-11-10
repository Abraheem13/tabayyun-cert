# Changelog

All design and code changes. Experiment results go in `EXPERIMENT_LOG.md`.

## [0.4.0] - 2026-11-10 (Day 4)
### Added
- `scripts/analyze_e02.py`: pre-registration scorecard (paired bootstrap, per-criterion best baseline, H7 prediction).
- `anchor_min_writers` ("k" | "f+1"); `tabayyun_anchored2`; `grow_anchored_basis` shared with baselines; `RobustAggMemory(basis="anchored")` (A11); `tabayyun_anchored2_add` (A3).
- Metric `installed`; per-query certificate at k* recorded.
- Pre-registration Amendment 1; Day 5 plan.

## [0.3.0] - 2026-11-10 (Day 3)
### Added
- `basis="anchored"`: key-anchored, growing, frozen basis (F13); `DirectionalDrift.resize`.
- Locality regime (`WorldConfig.support_min/max`, E02 `--support minority`), magnitude-aware metrics (`l2_recall`, `l2_flip`, `amnesia`).
- E02: same white-box adversary for the robust-aggregation baselines (`RobustAggMemory.delta_hook`), parallel workers, `--seed_offset`, `--alpha_q`, `--n`, `--tag`.
- `paper/proofs.md`: Lemma 1, Lemma 1′, Prop. 0.
- 7 new tests (58 total).
### Changed
- α^Q = 0.005 (clean-tuned). Pre-registration v1 frozen (tag `prereg-v1`), confirmatory seeds 100+.

## [0.2.0] - 2026-11-10 (Day 2)
### Added
- `certify.mean_field_half_life` (Lemma 1′) and its unit test.
- E01: orthogonal vs random streams, divergence guard, `--part/--kinds/--streams/--tag`.
- E01 rayn: fact-flip protocol under per-write norm filters.
- `scripts/run_day.py` (Windows), Day 3 plan.
### Fixed
- MLP memory divergence: unit-norm keys/values in E01 (as in Titans).
- Rayn metric: cosine replaced by flip of an established fact.
### Changed
- H1 restated: Lemma 1 is an upper bound; Lemma 1′ is the predictive law; momentum shortens persistence in busy memories.

## [0.1.0] - 2026-11-08 (Day 0)
### Added
- Core library, testbed, attack suite, baselines, run registry, 50 unit tests, E01/E02 scripts.
- `docs/proposal_v2.md`: 12 fixes to proposal v1 (F1–F12).
### Changed (relative to proposal v1)
- Gate statistics over active sources / corroborators only (F1).
- Runtime-computable certificate, Prop. R (F2).
- EMA promotion rule by default; v1 additive rule kept for ablation (F3).
- Linear certified promoted tier over any (deep) quarantine (F4).
- Basis frozen after round 1 (F5).
- Pairwise, per-query action certificate (F6).
- H1 on the exact persistence curve, measured with counterfactual twins (F7).
