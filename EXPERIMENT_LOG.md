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

---

## Day 5 (2026-11-10): confirmatory test of Amendment 1 (fresh seeds 110–119)

**Runs (CPU, 19 workers, commit `f65e3dc`):** dense 9180 runs (226 s), locality 6120 runs (165 s), plus scorecards generated on the RTX machine. Method = `tabayyun_anchored2`.

### Scorecard (Amendment 1, confirmatory)
| Criterion | Dense (n=10) | Locality (n=20) |
|---|---|---|
| P1 (≥ 50% ASR reduction vs Titans) | **9/9** | **6/6** |
| P2 (vs best robust baseline) | **7/9** | **6/6** |
| A2 (gate needed) / A6 (anchoring needed) | – | **6/6 / 6/6** |
| **H7 (Thm 3(b) predicts recall, ±0.10)** | **9/9**, max error 0.02 | **6/6**, max error 0.02 |
| H3 certificate violations | **0 / 2700** | **0 / 1800** |

- **Locality (main claim):** Tabayyun Recall_L2 0.54–1.00 vs **0.00 for every robust baseline in every cell**. Worst-case amnesia 0.0–0.5 vs 1.0. Krum gets its target **installed** in 40–100% of attacked runs; Titans in 100%.
- **`installed` (secondary): Tabayyun = 0.00 in all 15 cells, both regimes.** The attacker's value is never actually written into the trusted memory. The non-zero "ASR" (0.1–0.2) in starved cells is the coin-toss artefact on empty reads predicted in Amendment 1.
- **Dense, the 2 P2 failures** (f/n = 0.3, ρ ≤ 0.4) are exactly the starvation cells. Recall 0.35 / 0.91 against Thm 3(b) predictions of 0.36 / 0.93. It is a **predicted, tunable cost**, reported as a limitation with its operating curve.
- **Day 4 vs Day 5:** anchoring v2.2 raises recall in the starved cells (dense 0.17 → 0.35; locality 0.32 → 0.54) and turns H7 from 13/15 into 15/15.

### Ablations (descriptive)
- **A11 fairness:** robust baselines with the same anchored basis still have locality Recall_L2 = **0.00** (median, TrMean, Krum), and Krum-anchored installs the attacker's value in 29% of runs. The majority requirement, not the basis, is the cause. In the dense regime they are fine (0.95–0.99).
- **A3:** the v1 additive promotion rule fails on utility (dense 0.11, locality 0.00) → EMA (F3) is required.

### Certificate: sound and exactly tight
- 0 violations in 4500 runs. An apparent per-query ratio of 6.9 was float32 noise at magnitudes around 1e−8. Over all runs with a certificate > 1e−4 the maximum ratio is **1.0000003**.
- The omniscient edge-of-range attacker reaches true/certified = **0.94–0.98 median, 0.998 max** (Frobenius). Weak attacks reach about 0.29.
- This motivated **Theorem 4 (instance-optimality)**: Prop. R is the smallest sound bound computable from the observed values. The proof is short and complete (`paper/proofs.md`).

### Status vs roadmap
Week-1 Tier-A goals are met early: H1, H3, H5, H6/P2-loc, H7, A2, A3, A6, A11 and Thm 4. **Gate 1 is effectively passed.** The criteria (0 violations; corroboration matters in the sparse/local regime; Tabayyun ≥ the best baseline against adaptive attacks) all hold.

**Next (Day 6):** port the 2026 memory defences as baselines: SMSR-style randomised ablation and voting, TMA-NM-style corroboration-gated elevation (writer count, no robust statistics), RobustRAG isolate-then-aggregate, and Miras-Yaad (Huber attentional bias). Run them on fresh seeds 120–129 (Amendment 2: P2 is recomputed against the extended baseline set).

### Day 6 preparation (same day): 2026 defences ported
- `HuberSharedMemory` (Miras-Yaad), `SMSRMemory` (randomised source ablation + median vote over 21 subsets of 30% of sources), `IsolateAggregateMemory` (RobustRAG), `tma_corr` (TMA-NM-style: ≥ k distinct writers, plain mean, no cone, no trimming, no certificate). 60/60 tests.
- **Exploratory probe (seeds 0–2, locality, ρ = 0.4), not results:**
  - TMA-NM-style corroboration: Recall_L2 0.94–1.00, **but ASR 1.00 and amnesia 1.00**.
  - SMSR: Recall_L2 0.00, ASR 0.67–1.00. RobustRAG: ASR 0.00, Recall_L2 0.00. Miras-Yaad: Recall_L2 0.96–0.99, ASR 1.00.
  - Tabayyun: Recall_L2 0.94–1.00, ASR 0, amnesia 0.
  - **Key insight for the novelty argument:** corroboration-gated elevation (the 2026 TMA-NM idea) is **necessary but not sufficient** for parametric memory. Once ≥ k sources corroborate, f of them can still steer the plain mean. The *robust statistics inside the corroborating set* (cone test + trimmed mean) and the certificate are what make it secure.
- Amendment 2 registered; Day 6 = confirmatory on seeds 120–129.

---

## Day 6 (2026-11-10): Amendment 2, strict comparison against the 2026 defences (seeds 120–129)

**Runs (CPU, commit `a7dc484`):** dense 8100 runs, locality 5400 runs, scorecard `results/e02/scorecard_amend2.json`, Pareto analysis `scorecard_amend2_pareto.json` (`scripts/pareto_e02.py`).

### As registered (Amendment 2)
| Criterion | Dense | Locality |
|---|---|---|
| P1 | 9/9 | **5/6** |
| P2 (per-criterion best of 13 baselines) | 7/9 | **0/6** |
| H7 (Thm 3(b) predicts recall) | 9/9 | 6/6 |
| H3 violations | 0/540 | 0/360 |

**Why locality P2 = 0/6: a flaw in the criterion I wrote for Amendment 2 (my error), reported as a fail.** Adding undefended single-tier memories (Titans + clip, Miras-Yaad) and TMA-NM-style corroboration to a "best per criterion" comparison means recall is compared against memories with **ASR = 1.00** (recall 0.94–1.00), while ASR is compared against RobustRAG / anchored median, which have **recall = 0.00**. No method can be 0.20 better in recall than an undefended memory, so the criterion was unsatisfiable by construction.

**Why P1 fails in locality f = 0.15, ρ = 0.2:** Tabayyun's registered ASR is 0.50, but **installed = 0.00**. This is the empty-read coin toss on facts that were not promoted (recall 0.55 = Thm 3(b) prediction 0.54), not an installed flip.

### Post-hoc multi-objective analysis (labelled post hoc in the paper)
Objectives: worst ASR, worst amnesia, worst installed, clean Recall_L2.
- **Locality: Tabayyun is Pareto-undominated in 6/6 cells** and dominates 9–14 of 14 baselines.
  - Best *secure* baseline (installed ≤ 0.1, ASR ≤ 0.2): **Recall_L2 = 0.00 in all 6 cells** vs Tabayyun 0.55–1.00.
  - Best *useful* baseline (recall within 0.05 of Tabayyun): TMA-NM-style corroboration, **ASR = 1.00** (installed up to 0.60).
- **Dense:** undominated in 7/9 cells. It is dominated by Krum / anchored median / RobustRAG only in the f/n = 0.3 starvation cells (ρ = 0.2, 0.4), exactly where Thm 3(b) predicts low recall.

### What the 2026 defences do on parametric memory (locality, worst case)
| Defence | Recall_L2 | ASR | installed |
|---|---|---|---|
| Miras-Yaad (Huber) | 0.94–1.00 | 1.00 | 1.00 |
| SMSR-style ablation + vote | 0.00 | 0.80–0.90 | 0.60–0.90 |
| RobustRAG-style isolate+aggregate | 0.00 | 0.00 | 0.00 |
| TMA-NM-style corroboration (count only) | 0.55–1.00 | 1.00 | 0.00–0.60 |
| **Tabayyun-Cert** | **0.55–1.00** | 0.00–0.50* | **0.00** |

*the coin-toss artefact on unpromoted facts (see above).

**Conclusion:** among 15 memories, Tabayyun is the only one that is both secure and useful when knowledge is local. Corroboration without robust statistics (the closest 2026 idea) is fully steerable.

### Amendment 3 (prospective; applies to Tier B/C and to the final Tier-A rerun on Day 27)
Registered in `docs/preregistration.md`. Primary security metric = **installed**, with flip-ASR secondary. Primary comparison = Pareto non-domination plus recall vs best *secure* baseline plus security vs best *useful* baseline.

### Note
The `prereg-v1-amend2` tag did not reach GitHub (network drop on the PI machine). It is re-pushed with the Day 8 command. Amendment 2's text is in commit `26aa550`, before the Day 6 runs (`a7dc484`).

**Decision: Tier A is complete.** Gate 1 is passed (0 certificate violations in 12,240 Tabayyun runs; locality advantage confirmed on 3 independent seed sets). Moving to **Tier B (learned memories, MQAR)**.

---

## Day 7 preparation: Tier B (trained memory LM on MQAR)

**Built**
- `models/memlm.py`, **MemLM**: embedding, full width-2 causal conv, MLP, then **one Titans linear-memory layer** with data-dependent gates. Logits = W_U(h + W_O y) are **affine in the memory read y**, so Thm 3(a) certifies the *predicted token* exactly with W = W_U W_O. All cross-token mixing happens in the memory layer, so per-source quarantine isolates sources completely.
- `data/mqar.py`: multi-segment MQAR (Zoology protocol: keys and values in disjoint vocab halves); one write per asserted fact.
- `e03_mqar_train.py` (curriculum: facts 4 → max over 40% of steps) and `e04_mqar_attack.py` (token-level attacks on the trained model; 11 defences; certified-token fraction; soundness check against the oracle reference).
- Tests: the model's scan equals the certified Titans `LinearMemory` (to 1e−10); logits are exactly affine in y. **62/62 pass.**

**Development findings (CPU, tiny model; not results)**
1. With writes at every context position, MemLM plateaus at ~32%. With **one write per asserted fact** (value positions) it reaches 96% in 1.2k steps.
2. Harder settings (12 facts × 4 segments) need a **curriculum**: 33% without, **98% with** (1.6k CPU steps).
3. E04 smoke (3 trials, tiny model): 0 certificate violations and 0 action-certificate violations. But the **certified-token fraction is low (0.00–0.07)**: the per-query bound is large relative to logit margins in a small model with correlated learned keys. Also, token-level argmax is less sensitive to magnitude dilution than Recall_L2, so the baselines lose less recall than in Tier A. **Both points must be checked on the GPU models before any Tier-B claim is registered.**

**Day 7 = GPU training of two models (d128/dk64, d256/dk128) plus exploratory attack runs (trials 0–99).** Confirmatory Tier-B criteria will be registered (Amendment 4) after seeing the exploratory numbers, with fresh trials 200+.

---

## Day 7 (2026-11-10): Tier B, trained memory LMs (RTX 5080), exploratory attacks (trials 0–99)

**Note:** the first Day-7 attempt did not run (the PI machine's `git pull` failed, so it had the old runner; the log has only a header, commit `b45f452`). The rerun is commit `45de15d`.

### Training (E03, `results/e03/`)
| Model | Params | Steps | Final MQAR accuracy | Time |
|---|---|---|---|---|
| MemLM d128 / dk64 | 0.20 M | 8000 | **99.98%** | 23.7 min |
| MemLM d256 / dk128 | 0.66 M | 8000 | **99.92%** | 24.6 min |

### Exploratory attack results (E04, 16 facts, 100 trials per cell × 5 attacks × 8 cells × 11 defences, 44k rows per model)
- **0 certificate violations and 0 certified-token violations** (88k rows).
- **Tabayyun has the lowest attacker-installed rate of any useful defence:** worst-case installed 0.00–0.16. Titans is 0.23–0.61, mean / TMA-NM-style about 0.50, TrMean 0.07–0.46, GeoMed 0.07–0.42, SMSR up to 0.48. Krum and RobustRAG install little but recall almost nothing (Krum 0.01–0.80, RobustRAG 0.02–0.96).
- **Certified tokens on real models:** dense f=1, ρ=0.7: **0.91–0.95** of tokens certified (clean), 0.88–0.91 under attack. Dense f=2, ρ=0.7: 0.47–0.56. **Locality: 0.00–0.09.**
- Dense recall is on par with TrMean/median at ρ=0.7 (1.00 vs 0.99 for f=1; 0.82–0.83 vs 0.98 for f=2). It starves at f=2, ρ=0.4 (0.41–0.45). That is the single-shot regime (one assertion opportunity per source, versus 10 rounds in Tier A), consistent with Thm 3(b): P(Bin(8, 0.4) ≥ 5) = 0.17.

### Diagnosis of the weak locality numbers
The activity threshold ν = 1e−3 came from Tier A, where fact keys are exactly orthogonal. With **learned keys every source has small cross-talk reads along every direction**, so non-supporters count as active, dominate the median and push the real supporters out of the cone. This also widens the certificate (more, and more dispersed, corroborators).

**Fix (v2.3): robust relative activity threshold** `nu_rel`. A source is active along u iff ‖D[s,u]‖ ≥ ν_rel · (the (f+1)-th largest norm along u). At most f values can exceed that order statistic, so it is ≤ the largest honest norm: the adversary cannot push every honest supporter below the bar. Unit test added (63/63).

### Next (Day 8)
**Clean-only tuning** of ν_rel ∈ {none, 0.1, 0.2, 0.3, 0.5} × γ ∈ {0.3, 0.5} on trials 0–99, attack = none. The objective was fixed in advance: clean recall + certified fraction. Then Amendment 4 registers the Tier-B confirmatory test (Amendment 3 metrics, trials 200–299). The two small checkpoints (~1 MB each) are released in `checkpoints/` for reproducibility.

---

## Day 8 (2026-11-10): clean-only tuning; the cross-talk finding

**PI run failed** (commit `14a58e4`): `SyntaxError: name 'DEFENCES' is used prior to global declaration` in E04. My bug (I edited the script without re-running it). Fixed in `2208d8a`. The checkpoints were released (`6c9017b`), so **the tuning was run on the released models on CPU** (`results/e04/tuning_clean.json`, 36 configs × 2 models, attack = none, trials 0–99, 10.5 min).

**Objective changed before any tuning data existed:** "certified fraction" rewards empty memories (an empty read trivially equals the empty reference). The objective is now recall + **certified-and-correct**.

### Result: my ν_rel hypothesis is falsified as a strict improvement
| model | best config (pre-set objective) | mean recall | mean certified-correct |
|---|---|---|---|
| d128 | ν_rel none, γ 0.3 | 0.70 | 0.30 |
| d256 | ν_rel none, γ 0.3 | 0.70 | 0.26 |

ν_rel raises certification (up to 0.43) but lowers recall (to 0.46). It is a trade-off, not a fix.

### Diagnosis (40 clean locality trials, f = 2, ρ = 0.4, d128)
| honest observers of the fact | promoted | corroborators counted | correct |
|---|---|---|---|
| 0 | (not anchored) | – | 0.00 |
| 1–2 | (not anchored: needs f+1 = 3 writers) | – | 0.01 |
| **3–4 (< k = 5)** | **0.97** | **8.2** | **0.93** |
| ≥ 5 | 1.00 | 10.7 | 1.00 |

**Cross-talk corroboration:** with learned, non-orthogonal keys, honest sources that never saw a fact have small reads along its direction and pass the cone test, so a fact seen by only 3–4 honest sources gets "k = 5 corroborators".
- This is not a security hole: the extra corroborators are honest, adversary-only keys are never anchored, and the certificate stays sound (0 violations).
- But it loosens the "k independent witnesses" semantics and explains why recall exceeds Thm 3(b).
- With ν_rel = 0.3 the method follows Thm 3(b) closely (exploratory, d128): dense f=1, ρ=0.4: 0.77 vs 0.77; dense f=2, ρ=0.7: 0.82 vs 0.81; locality f=2, ρ=0.7: 0.58 vs 0.59.

### Design for the Tier-B confirmatory test (Amendment 4)
- **Primary** = pre-set tuning winner (ν_rel none). **Secondary** = strict (ν_rel 0.3), on which H7 is tested.
- Added **observation rounds R ∈ {1, 3}**: single-shot is harsher than real memory streams (exploratory smoke, locality R = 3: strict certifies 83% of tokens as correct at recall 0.93).
- New scorer `scripts/analyze_e04.py`. An 8-trial exploratory preview predicts C3 (Amendment 3's −0.50 installed margin) will fail in Tier B. This expectation is written into Amendment 4 *before* the run, and the criterion is not changed.

**Day 9 = confirmatory Tier B (trials 200–299, both models).**

---

## Day 9 (2026-11-10): Tier-B CONFIRMATORY (Amendment 4; trials 200–299; both models)

**Runs (RTX 5080, commit `0118a8b`):** 96,000 rows per model (16 cells × 5 attack settings × 100 trials × 12 defences), 37 min per model. Scorecards in `results/e04/e04_{d128,d256}_confirm_scorecard.json`.

### Scorecard (both models identical except where noted)
| Criterion | primary `tabayyun` | secondary `tabayyun_strict` |
|---|---|---|
| **H3**: certificate + certified-token violations | **0** (192k rows) | **0** |
| **P1**: installed ≤ 0.5 × Titans | **32/32** | **32/32** |
| **C1**: Pareto non-dominated by all 9 baselines | **32/32** | **32/32** |
| **C2** (locality): recall ≥ best secure baseline + 0.20 | 10/16 (5/8 + 5/8) | 3/16 (1/8 + 2/8) |
| **C3**: installed ≤ best useful baseline − 0.50 | 0/32 (*pre-declared expected failure*) | 0/32 |
| **H7**: Thm 3(b) predicts recall (±0.10) | 18/32 (not a criterion; cross-talk) | **32/32** (most within 0.03) |

### What holds
- **Security everywhere.** Worst-case installed rate is 0.00–0.07 in 30/32 cells (max 0.22 at locality f=3, ρ=0.4, R=1), against Titans 0.13–0.56. Among baselines with comparable recall, installed rates are 0.02–0.42 (e.g. locality f=3, ρ=0.7, R=1: median recall 0.83–0.86 with **installed 0.28–0.30**, vs Tabayyun recall 0.81 with **installed 0.03**).
- **Never dominated.** In all 64 (method × model × cell) comparisons no baseline is at least as good on installed, amnesia and recall.
- **Certified tokens.** Dense with R = 3: **0.58–1.00** of all tokens are certified *and* correct (0.90–1.00 in 6/8 cells). Strict variant in locality R = 3: 0.30–0.81. Primary variant in locality: 0.00–0.33.
- **Theory predicts the cost of security** for the strict variant in every one of 32 cells.

### What does not hold (reported as such)
- **C2 fails in 6/16 (primary) and 13/16 (strict) locality cells.** Mostly at R = 3, where facts are observed many times and the best *secure* baseline (GeoMed / TrMean, installed ≤ 0.1) reaches recall 0.90–0.98. **The large Tier-A recall advantage over robust aggregators does not fully carry over to token-level MQAR.** Argmax decoding tolerates the dilution that majority aggregators cause, whereas Tier A's magnitude-aware Recall_L2 does not. Tier B's advantage over robust aggregators is therefore **security (installed) plus certification at comparable recall**, not recall.
- **C3 fails everywhere**, as written down in Amendment 4 before the run.
- The strict variant pays in recall when observation is single-shot (R = 1, locality: 0.04–0.61), exactly as Thm 3(b) predicts.

### Paper message after Tier B
1. On a trained memory LM, Tabayyun is the only defence that is **never Pareto-dominated**, and it halves or better the attacker's success against Titans in every cell.
2. When sources re-observe facts (R = 3), the strict variant **certifies up to 100% of predicted tokens**, and its utility is predicted in closed form.
3. Robust aggregators are competitive in token recall when facts are well observed, but they are attacker-steerable (installed up to 0.42) and certify nothing.

**Next:** adaptive gradient attacker (PGD on a differentiable relaxation of the gate, plus BPDA; Day 13 of the roadmap, pulled forward), then federation (Days 15–16) and O-RAN (Days 17–18). While I build the attacker, the GPU trains a larger model (d512) for the scaling ablation.

---

## Day 10 (2026-11-11): d512 model; adaptive gradient attacker built and pre-registered

**d512 / dk256 (2.36 M params): MQAR 99.98%**, checkpoint released. **But training took 21.3 h** (76,785 s), against 24 min for d128. My estimate (30–40 min) was wrong. Likely cause: the scan's autograd graph (B·d_v·d_k·L ≈ 128·256·256·300 floats per state tensor) exceeds the 16 GB of VRAM, and the Windows driver silently falls back to shared system memory. **Action before any larger model:** chunked scan with gradient checkpointing, or a smaller batch with gradient accumulation. Logged as a known compute issue.

**E05 adaptive attacker** (`experiments/e05_adaptive.py`): PGD/BPDA with unbounded white-box control of the adversarial sources' memory states, CW token loss, 3 restarts, run through each defence's exact forward pass. Smoke (trials 0–1, debugging only): PGD installs **0.25–1.00 against TrMean / median / GeoMed** (the static edge-of-range attack installed 0.00 on them), and 0.25 against primary Tabayyun in one locality cell. Strict: 0.00. 0 certificate violations.

**Amendment 5 registered before the confirmatory run** (trials 300–329, both models; H8a/b/c). The run executes on CPU in this environment from the released checkpoints (≈ 25 min per model).

---

## Day 11 (2026-11-12): federation confirmatory (Amendment 6), PASS 4/4

E06, trials 400–419, 8 rounds, 5,600 runs, 960 s on 19 workers. N = 12 clients × 7 sources (1 adversarial source inside every client); f_c ∈ {1, 2} Byzantine clients send crafted rows **and under-report their own bounds**. Scored by `scripts/analyze_e06.py` → `results/e06/scorecard_amend6.json`.

| Criterion | Result |
|---|---|
| H9a: 0 Theorem 5 violations | **8/8 cells, 0 violations in 800 fed_tabayyun runs** |
| H9b: worst installed ≤ 0.05 and worst flip ≤ 0.10 | **8/8** (all 0.00) |
| H9c: never Pareto-dominated | **8/8** |
| H9d: FedAvg-Titans worst flip ≥ 0.5 | **8/8** (1.00 everywhere, installed 1.00) |

**Reading.**
- **Federated Tabayyun:** clean recall 1.00, worst flip 0.00, worst amnesia 0.00 in every cell.
- **FedAvg (plain mean) of Tabayyun clients:** flip 1.00 in every cell. The server mean is the weak link even when every client is protected.
- **Dense regime:** robust server aggregators (TrMean, median, GeoMed, Krum) tie with Tabayyun.
- **Locality regime:** robust server aggregators **erase minority-held facts**.
  - Clean recall 0.68–0.77 at f_c = 1.
  - Worst amnesia 0.45–0.85 at the attacked key.
  - This reproduces the Tier-A locality finding one level up: majority aggregation over clients deletes facts known to a few clients.
- Krum flips 0.05 in two locality cells.

---

## Day 12 plan (Amendment 7, registered before the runs)
1. **E07 ablations** (A4 momentum in M^P, A5 ν/γ/cone/k/τ/α_P, A10 after-promotion policy), seeds 500–509, CPU, about 2 min. A4 adds `eta_P` to `TabayyunConfig` (default 0, so behaviour is unchanged). Smoke result: η_P = 0.9 gives drift/cert ≈ 2.1, i.e. the certificate fails without the momentum-free design, as predicted.
2. **d512 scaling replication** of Amendment 4 (trials 200–299, GPU inference only, so no VRAM spill expected).

---

## Day 10/11 addendum: E05 adaptive-attack confirmatory (Amendment 5), PASS on both models

PGD/BPDA on the attackers' memory states (CW token loss, 100 steps, 3 restarts), trials 300–329, 16 cells × 7 defences per model. CPU runtime: d128 about 45 min, d256 4,465 s. Scored by `scripts/analyze_e05.py`.

| | d128 | d256 |
|---|---|---|
| H8a: certificate and certified-token violations | **0** | **0** |
| H8b: strict PGD-installed ≤ 0.10, every cell | **16/16** (worst 0.033) | **16/16** (worst 0.083) |
| H8c: strict ≤ min(TrMean, median, GeoMed) + 0.05 | **16/16** | **16/16** |
| Worst PGD-installed: TrMean / median / GeoMed | 1.00 / 0.40 / 1.00 | 1.00 / 0.48 / 1.00 |
| Worst PGD-installed: Krum / RobustRAG | 0.03 / 0.10 | 0.03 / 0.13 |
| Primary `tabayyun` (descriptive) | 12/16 ≤ 0.10, worst 0.32 | 12/16 ≤ 0.10, worst 0.42 |

**Reading (honest).**
- The gradient attacker **breaks TrMean and GeoMed completely**, even though the static edge-of-range attack did not. Robustness to hand-crafted attacks was overstated for those baselines.
- **Strict Tabayyun holds** against the adaptive attacker.
- **The primary variant does not hold**: up to 0.42 installed, in locality cells with R = 1. The cause is the cross-talk corroboration diagnosed on Day 8: adversarial writes leak into directions that honest sources corroborate. The certificate is still sound in those runs (0 violations), because it bounds drift from the honest-corroborator reference and the leak passes through honest reads. **Consequence for the paper:** strict (ν_rel = 0.3) is the recommended secure configuration; primary is the high-recall setting.
- Krum and RobustRAG also resist PGD here. Their cost is utility: Krum keeps one source and RobustRAG has lower recall (Tier B Day 9). Both certify nothing.

---

## Day 12 (2026-11-12): Amendment 7: d512 scaling replication + E07 ablations

**Git note.** The first two pushes did not reach GitHub because the local repository was in a detached-HEAD state left by an interrupted `pull --rebase`. Fixed with `git checkout -B main`.

### (a) d512 replication of Amendment 4 (trials 200–299, 96,000 rows, 2,293 s on the RTX 5080: inference only, no VRAM spill)
| Criterion | primary | strict | (d128 / d256, Day 9) |
|---|---|---|---|
| H3: certificate + certified-token violations | **0** | **0** | 0 / 0 |
| P1 (better than Titans) | **16/16** | **16/16** | 32/32 |
| C1 (never Pareto-dominated) | 15/16 | 15/16 | 32/32 |
| H7 (Thm 3b predicts recall) | 9/16 (not a criterion) | **16/16** | strict 32/32 |
| C2 (locality recall) | 3/8 | 0/8 | 10/16, 3/16 |
| C3 | 0/16 | 0/16 | 0/32 (expected) |

- **New failure: C1 in one cell** (dense, f = 2, ρ = 0.4, R = 3). RobustRAG dominates both variants: recall 0.98 vs 0.92, both with installed 0.00. Reported as a fail. RobustRAG certifies nothing; Tabayyun certifies-and-is-correct on 0.68–0.70 of tokens in that cell.
- **C2 is weaker at d512** (3/8 and 0/8, against 10/16 and 3/16 at d128/d256). The token-level recall advantage over robust aggregators shrinks further with model size. The Tier-B claim stays **security plus certification at comparable recall**, not recall.
- Certified-and-correct tokens: dense 0.96–1.00 at R = 3, ρ = 0.7. Strict locality R = 3: 0.29–0.84.
- **The scaling conclusions replicate:** 0 violations, P1 everywhere, and the strict variant's utility is predicted by Thm 3(b) in every cell.

### (b) E07 ablations (3,200 runs, 211 s; seeds 500–509; descriptive). Full rows in `ABLATIONS.md`.
| variant | recall dense | recall locality | worst installed | worst amnesia | violations | max drift/cert | certificate (Frobenius) |
|---|---|---|---|---|---|---|---|
| base | 1.00 | 0.99 | 0.00 | 0.00 | 0 | 1.00 | 0.23 |
| η_P = 0.5 (A4) | 1.00 | 0.99 | 0.00 | 0.00 | **40** | **1.45** | 0.23 |
| η_P = 0.9 (A4) | 0.91 | 0.41 | 0.00 | 0.70 | **40** | **2.13** | 0.23 |
| no cone (A5) | 1.00 | 0.99 | 0.00 | 0.00 | 0 | 1.00 | **4.62** |
| k + 2 (A5) | 1.00 | 0.58 | 0.00 | 0.60 | 0 | 1.00 | 0.18 |
| reset (A10) | 1.00 | 0.61 | 0.00 | 0.90 | 0 | 1.00 | 0.23 |
| decay (A10) | 0.95 | 0.82 | 0.00 | 0.80 | 0 | 1.00 | 0.68 |
| ν, γ, τ, α_P sweeps | 1.00 | 0.97–1.00 | 0.00 | ≤ 0.10 | 0 | 1.00 | 0.20–0.24 |

**Predictions vs outcome (Amendment 7b):**
- **A4 confirmed.** The momentum-free promoted tier is necessary for the certificate. Even η_P = 0.5, which costs no utility, violates it in 40 runs. This motivates the design and is good for the paper.
- **A10 confirmed.**
- **k + 2 confirmed.**
- **No cone: prediction wrong.** I predicted higher installed. Installed stays 0 (the trimmed mean already removes the edge values), but the certificate becomes 20× looser. The cone test is a **tightness** component, not a security one.
- ν, γ, τ and α_P are inert in Tier A, so the method is not fragile to these hyper-parameters.
