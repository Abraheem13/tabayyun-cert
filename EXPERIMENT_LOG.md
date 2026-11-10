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

---

## Day 3 (2026-11-10): E01 completed, new regime design, pre-registration frozen

**Runs (RTX 5080, commit `ab49140`):** `results/e01/e01_halflife_mlp_random.json` (1072 s), `e01_halflife_linear_orth.json` (358 s), `e01_rayn.json` (89 s). 51/51 tests passed.

### H1 persistence: confirmed
- **Linear, orthogonal stream:** measured = Lemma 1 **exactly** in 42/42 cells (zero error, all seeds). The v1 formula is off by up to 2.6×.
- **Linear, random stream (Day 2):** Lemma 1′ within −9%…+4%, r = 0.9996.
- **MLP, random stream:** 0 divergences after the fix. Lemma 1 upper-bounds 41/41 finite cells. One cell (η = 0.95, α = 0.001) has 2/5 seeds whose poison **never halves within 3000 steps**: a deep memory can retain poison indefinitely at high momentum and tiny forget. This is reported as a finding. MLP persistence is about 0.5× the linear one, because the forget gate scales both layers (α_eff ≈ 2α). With α_eff = 2α and one interference constant fitted on the 6 η = 0 cells, the mean-field law predicts the other 35 cells with r = 0.994 and median ratio 1.00 (exploratory).

### H5 necessity: confirmed (10 seeds × η ∈ {0, 0.9})
| per-write filter G | single-shot ASR (end) | rayn ASR (end) | rayn median writes to flip | rayn max write norm |
|---|---|---|---|---|
| none | 0.00 (transient flip only) | **1.00** | 28 | 0.010 |
| 0.5 | 0.00 | **1.00** | 27.5 | 0.010 |
| 0.2 | 0.00 | **1.00** | 21.5 | 0.009 |
| 0.1 | 0.00 | **1.00** | 25.5 | 0.0045 |
| 0.05 | 0.00 | **1.00** | 33.5 | 0.0023 |
| 0.02 | 0.00 | **1.00** | 35 | 0.0009 |

Writes-to-flip barely depend on G, because the filter clips the honest refresh too. That is the core of **Prop. 0 (proof sketch in `paper/proofs.md`)**: norm filters turn the contest into one of write frequency, which the attacker controls. Momentum helps the attacker (η = 0.9: 11 writes unfiltered vs 45 at η = 0).

### Design finding: a fixed basis mixes facts → key-anchored basis (v2.1)
With the identity basis, every basis direction mixes many facts, so honest sources that saw different facts "disagree", and the gate does not really check per-fact corroboration. New `basis="anchored"`: a key becomes a frozen basis direction once ≥ k distinct sources wrote it (in the last 5 rounds). The cluster centre is a coordinate median, orthogonalised against the existing directions. Existing directions never change, so the per-direction certificate stays valid; new directions start with zero error. Tests: certificate soundness with the anchored basis (6 new cases), and fact keys recovered while adversary-only keys are ignored. **58/58 tests pass.**

### Exploratory probes (seeds 0–2, CPU, not results): why the regime matters
1. **Dense regime** (n = 10, every source can see every fact, f/n = 0.2): median, Krum and TrMean also reach ASR 0. Tabayyun ties on security and adds only the certificate. **The v1 H2 claim (≥ 25% better than the best robust baseline) would have failed here.**
2. **Clean-only tuning:** the magnitude-aware Recall_L2 showed that α^Q = 0.02 shrinks stored values for every quarantine method. α^Q = 0.005 gives clean Recall_L2 = 1.00 for all of them → adopted.
3. **Locality regime** (n = 20, each fact known to only k…(n−f)/2 honest sources), 648 runs:
   - Tabayyun-anchored: worst-attack ASR 0, amnesia 0, Recall_L2 0.91–1.00 (clean and attacked).
   - mean / TrMean / median / GeoMed / Krum: **clean Recall_L2 = 0.00**. They erase every minority-held fact, since they need a global majority. Worst-case ASR goes up to 0.67 (median, GeoMed) and 1.00 (Krum, TrMean at f = 3).
   - Titans: ASR 1.00. Tabayyun-identity (A6): Recall_L2 0.03–0.13. Tabayyun-noCorr (A2): fails. **Both new components are necessary.**
   - 0 certificate violations / 216 runs. Max true-drift / certificate = 1.0000 (float32), so the certificate is **attained** by the edge-of-range attacker: evidence for tightness (Thm 4).
   - Caveat to test later: give the baselines the anchored basis too (expected not to help them, since they still need a majority per direction). Added to Day 6.

**Thesis sharpened:** *robust aggregation protects against a minority by requiring a majority, so it also silences minority-held truths. Corroboration needs only k independent sources.* This matches the O-RAN reality that most facts are local to a few neighbouring cells.

### Pre-registration frozen (`docs/preregistration.md`, tag `prereg-v1`)
Regimes, metrics (Recall_L2, L2 flip, amnesia), success criteria (P1, P2-dense non-inferiority, P2-loc/H6, A2, A6, H3) and **confirmatory seeds 100–109** are fixed before any confirmatory run.

**Decision:** Day 4 = confirmatory E02, dense (6480 runs) and locality (4320 runs), on CPU workers.

---

## Day 4 (2026-11-10): first confirmatory E02 (prereg-v1, seeds 100–109)

**Runs (CPU, 19 workers, commit `7b01795`):** dense 6480 runs (81 s), locality 4320 runs (62 s). Scored by `scripts/analyze_e02.py`, giving `results/e02/prereg_scorecard.json`. Paired bootstrap, 10k resamples.

### Scorecard (method = `tabayyun_anchored`, the registered method)
| Criterion | Dense (n=10) | Locality (n=20) |
|---|---|---|
| **P1** (≥ 50% ASR reduction vs Titans) | **9/9 PASS** | **6/6 PASS** |
| **P2** (dense: non-inferior to best robust baseline; locality: +0.20 recall, −0.20 amnesia, ≤ ASR) | **6/9** | **5/6 as registered, 6/6 with fp tolerance (Amendment 1)** |
| **A2** (corroboration gate needed) | – | 5/6 as registered, 6/6 with fp tolerance |
| **A6** (anchored basis needed) | – | **6/6 PASS** |
| **H3** (certificate violations) | **0 / 1620** | **0 / 1080** |

**Locality, the main claim (worst case over 5 attacks, 10 seeds):** Tabayyun-anchored has ASR 0.00–0.20, amnesia 0.00–0.50 and clean Recall_L2 0.32–1.00. **Every robust baseline (mean, TrMean, median, Krum, GeoMed, CClip) has clean Recall_L2 = 0.00 and amnesia = 1.00 in every cell.** They erase all minority-held facts, and Krum is also flipped (ASR 0.8–1.0). Differences vs the best baseline: recall **+0.32 to +1.00**, amnesia **−0.50 to −1.00**, ASR −0.50 to 0.00.

**Dense failures (honest):**
- f/n = 0.2, ρ = 0.2: recall difference 0.00 [−0.03, +0.03]. Non-inferiority within 0.02 is not established (underpowered at 10 seeds).
- f/n = 0.3, ρ = 0.2 and 0.4: **real failures.** Recall 0.17 / 0.88 vs Krum 0.95 / 1.00. With f = 3 we need k = 7 corroborators out of only 7 honest sources, so every honest source must see the fact. Under sparse observation, corroboration starves.

### Why it fails where it fails: Theorem 3(b) predicts it (H7, exploratory on Day 4)
Clean recall equals the probability that a fact gets k corroborators in time, P(Bin(m, 1−(1−ρ)^{R−1}) ≥ k). For the registered method this holds 8/9 (dense) and 5/6 (locality) within ±0.10. The misses come from the v2.1 anchoring rule (k writers inside a 5-round window), an extra bottleneck the theorem does not model.

### Diagnosis and fix (exploratory probes, seeds 0–5)
- **`tabayyun_anchored2`**: anchor a key once f+1 distinct sources wrote it (at least one honest); promotion still needs k. Thm 3(b) then predicts recall **within 0.07 in 8/8 cells** (e.g. dense f/n=0.3, ρ=0.2: 0.38 observed vs 0.36 predicted; ρ=0.4: 0.93 vs 0.93).
- **Fairness (A11):** robust baselines given the same anchored basis still have Recall_L2 = 0 in the locality regime. Their failure comes from the majority requirement, not from the basis.
- **A3:** the v1 additive promotion rule fails on utility (Recall_L2 = 0), which confirms fix F3 (EMA).
- **Metric caveat:** when a fact is never promoted the read is ≈ 0, and the registered ASR (v* vs v_true comparison) becomes a coin toss, so some "ASR" in starved cells is really amnesia. Added the secondary metric `installed`.

### Paper message after Day 4
1. Where facts are local (the realistic O-RAN case), Tabayyun dominates every robust aggregator on recall and amnesia (+0.3 to +1.0) at equal or lower ASR, with zero certificate violations.
2. Where every source sees every fact, Tabayyun ties the best robust aggregator on security and adds a certificate.
3. Its utility cost is a closed-form operating curve (Thm 3(b)) that the operator controls through (k, τ). It is largest when f/n → 1/3 under sparse observation. **The paper reports this regime as a limitation, with the curve.**

**Decision:** Amendment 1 registered (`docs/preregistration.md`). **Day 5 = confirmatory anchored2 + H7 + A11 + A3 on fresh seeds 110–119**, with scorecards generated on the RTX machine.
