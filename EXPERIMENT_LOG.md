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

---

## Day 1 (2026-11-10): GPU environment

**Machine:** Windows 10, Python 3.11.9, torch 2.11.0+cu128, **NVIDIA GeForce RTX 5080** (capability 12.0, sm_120 in the arch list, 17.1 GB). bf16 matmul **101.7 TFLOP/s**. The environment is valid for the whole plan.

**Tests:** 50/50 pass on Windows. pytest's faulthandler printed "Windows fatal exception: access violation" traces inside `test_hull_and_runtime_certificate_fuzz`, but the test passed. These are first-chance SEH exceptions inside torch's CPU kernels, which Windows reports even when they are handled. Fixed the output noise with `addopts = "-p no:faulthandler"`. A real crash would still kill the process and fail the run.

**Reproducibility check:** the GPU smoke numbers equal the Day-0 CPU smoke numbers to float32 precision (E01 log-log correlation 0.9518; rayn cos 0.9947 at 0.100× norm; E02 tightness ratios 4.59 / 3.93 / 1.19). Seeded generators make runs reproducible across devices and OSes. This is good evidence for the artefact.

**Issue:** `git push` was denied (403) because Windows git is logged in as a different GitHub account than the repo owner. Fix in the reply to the PI.

**Decision:** go to Day 2 (full E01 grid: linear + MLP, 7 η × 6 α × 5 seeds).
