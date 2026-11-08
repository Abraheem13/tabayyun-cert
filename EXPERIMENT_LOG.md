# Experiment log

One entry per day. Each entry: what ran (command + commit), key numbers (from `results/registry.jsonl`), interpretation, decision.
Smoke runs are **not results** and are never quoted in the paper.

---

## Day 0 (2026-11-08): setup

**Commit:** see `git log` (initial commit).

**Done**
- Core library: `memory.py` (linear + MLP Titans memory), `gate.py` (tabayyun gate v2 + FL aggregators), `tiered.py` (Tabayyun memory, EMA/additive rules, oracle reference), `certify.py` (Lemma 1, Thm 2′, Prop. R, Thm 3), `attacks.py`, `baselines.py`, `data/streams.py`, `metrics.py`, `registry.py`.
- `pytest`: **50 passed** (CPU).

**Smoke observations (2 seeds, not results)**
1. The v1 half-life formula ln2/(−ln max(η,1−α)) **does not capture momentum** when η < 1−α. With α=0.05 it predicts 13.5 steps for η ∈ {0, 0.5, 0.9}, while the exact Lemma-1 curve gives 14 / 16 / 25. → H1 restated on the exact curve (proposal v2, F7).
2. The first half-life measurement was confounded by benign interference on k*. Replaced by a counterfactual twin (poisoned vs clean memory, same stream). Log-log correlation between the exact prediction and measurement: 0.95 (smoke).
3. With random (non-orthogonal) benign writes, measured persistence is **shorter** than Lemma 1, by up to 2× at η = 0.9. That is consistent with Lemma 1 being an upper bound, and goes into the paper.
4. Rayn reaches cos(M k*, v*) = 0.99 at **10% of the single-shot per-write norm**, while single-shot only reaches 0.35 → early support for H5 / Prop. 0.
5. E02 smoke (f/n = 0.2, ρ = 0.7): Titans ASR 1.0; quarantine + mean, quarantine + TrMean and Tabayyun all reach ASR 0; certificate sound in all runs. **This regime is too easy to separate robust baselines from Tabayyun.** Day 4 must sweep ρ (sparse observation) and use the adaptive and unbounded attackers. That is where the corroboration gate should matter (H6).

**Decision:** proceed to Day 1 (GPU environment) and Day 2 (full E01).
