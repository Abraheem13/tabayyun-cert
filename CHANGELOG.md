# Changelog

All design and code changes. Experiment results go in `EXPERIMENT_LOG.md`.

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
