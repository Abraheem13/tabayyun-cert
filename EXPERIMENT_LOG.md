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

---

## Day 2 (2026-11-10): E01 full persistence grid

**Run:** `run_day.py 2` on the RTX 5080, commit `c9fabf5`, 1406 s. Output in `results/e01/e01.json` (old format) and `results/day02/console.txt`.

### Linear memory (42 cells × 5 seeds, random benign stream): **valid, strong result**
- **Lemma 1 is an upper bound in 42/42 cells.** It is tight (≤ 10%) when the forget gate dominates (α ≥ 0.02, η ≤ 0.5).
- At small α, persistence is far shorter than Lemma 1 (α=0.001, η=0: 272 vs 693 steps). Benign writes with random keys **self-correct** the poisoned direction at rate λ = 2θ/d_k, which acts like an extra forget gate.
- **New law (Lemma 1′, mean field):** S_t = ηS_{t−1} − λx_{t−1}, x_t = (1−α)x_{t−1} + S_t with λ = 2θ/d_k. It predicts **all 42 cells within −9%…+4%, log-log correlation 0.9996.** For comparison, the v1 formula has correlation 0.87 and errors up to 15×. Added `certify.mean_field_half_life` and a unit test.
- **Surprise: in a busy memory, momentum *shortens* poison persistence.** At α=0.001 the half-life is 272 steps at η=0 and 32 at η=0.95, because momentum amplifies the self-correction as well. v1 claimed "momentum lengthens persistence", which holds only on quiet (orthogonal) streams. **Consequences:**
  1. In H1, "momentum lengthens persistence" is true only for low-traffic keys. That is exactly the case for rarely-queried, high-value facts, so the attacker should target those.
  2. The momentum-free promoted tier must be justified by the **certificate** (Thm 2's 1/(1−η) factor), **not** by persistence. Proposal §3 rationale is updated.
  3. Attackers beat self-correction by **re-writing** (rayn), which motivates H5.

### MLP memory: **invalid, rerun on Day 3**
- At η ≥ 0.7, MLP memories diverged to ∞. The reported 0.0 / ∞ half-lives are artefacts.
- Cause: benign values were not normalised (‖v‖ ≈ 8 for d_v = 64), unlike Titans, which normalises them. Verified: with unit-norm values the MLP is stable even at η = 0.95, α = 0.001 (3000 steps).
- Fix: unit-norm keys and values everywhere, plus a divergence guard (diverged runs are counted, never averaged). Linear results are unchanged by this, since the twin difference is independent of the values.

### Rayn (H5): metric was flawed, redesigned
- Day-2 numbers: rayn cos 0.995 at 10% of the single-shot norm, single-shot cos ≈ 0. But cosine ignores magnitude: even a tiny clipped write into an empty direction scores cos ≈ 1.
- New protocol: **flip an established honest fact** (60 honest pre-writes, honest refresh every 25 steps). Success means the read projects more on v* than on v_true. A per-write gradient-norm filter G is swept.
- Smoke (2 seeds): with filter G = 0.05, **single-shot ASR 0, rayn ASR 1.0 at 0.2% of the unfiltered single-shot norm, about 33 writes to flip**, despite honest refresh. This is direct support for Prop. 0 / H5. The full sweep runs on Day 3.

**Decision:** Day 3 = MLP rerun (random stream), linear orthogonal stream (pure Lemma-1 check), rayn sweep (6 filters × 2 momenta × 10 seeds). Estimated 30–40 min.
