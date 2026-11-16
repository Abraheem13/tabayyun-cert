# Pre-registration v1 (FROZEN 2026-11-10, end of Day 3, tag `prereg-v1`)

Once tagged, this file only changes by appending a dated **amendment** with a reason. Results that contradict a hypothesis are reported, never dropped.

## How this was set
* Exploratory probes on Days 1–3 (seeds 0–9, written up in `EXPERIMENT_LOG.md`) were used **only** to design the regimes, metrics and the clean-tuned hyper-parameter α^Q. They are never quoted as results.
* **Confirmatory runs use seeds 100–109 (Tier A) and 200–204 (Tiers B/C), which no probe has touched.**
* Hyper-parameters are tuned on clean (attack = none) runs only: α^Q = 0.005 (Day 3), θ = 0.5, α^P = 0.5, τ = 50, γ = 0.3, ν = 0.05, k = 2f+1, β = f.

## Metrics (fixed)
* **ASR (flip)**: the read at k* is closer to v* (= −v_true) than to v_true, both in cosine (`asr`) and in L2 (`asr_l2`). The worst case over attacks is reported.
* **Amnesia**: ‖read − v_true‖ ≥ 0.5‖v_true‖ at the attacked key.
* **Recall_L2** (primary utility metric): fraction of benign facts with ‖read − v‖ < 0.5‖v‖. It is magnitude-aware, so dilution counts as loss. Nearest-cosine recall is reported as a secondary metric.
* **Certificate violation**: true drift > certificate·(1+1e−4)+1e−5.
* **Worst attack**: the maximum over {targeted, single_shot, rayn, edge_of_range, unbounded}, using the same omniscient adversary for every quarantine-based defence.

## Regimes (Tier A, E02)
* **Dense**: n = 10 sources, every honest source can observe every fact, f/n ∈ {0.1, 0.2, 0.3}, ρ ∈ {0.2, 0.4, 0.7}.
* **Locality**: n = 20 sources, each fact is known to m ~ U[k, ⌊(n−f)/2⌋] honest sources (a minority), f/n ∈ {0.1, 0.15}, ρ ∈ {0.2, 0.4, 0.7}.

Defences: Titans (+clip, +TTL); quarantine + {mean, TrMean, median, Krum, GeoMed/RFA, CClip}; Tabayyun-noCorr (A2); Tabayyun-identity-basis (A6); **Tabayyun-anchored (the method)**.

## Hypotheses and success criteria (cell = defence × f × ρ, averaged over 10 seeds; CIs from a paired bootstrap)
| ID | Regime | Claim | Success criterion |
|---|---|---|---|
| **P1** | dense and locality | Tabayyun-anchored vs Titans | worst-attack ASR reduced by ≥ 50% (relative) in every cell |
| **P2-dense** | dense | Tabayyun-anchored vs best robust baseline | **non-inferiority**: worst-attack ASR ≤ best baseline + 0.05, and clean Recall_L2 ≥ best baseline − 0.02. *We expect parity here; Tabayyun's added value in this regime is the certificate.* |
| **P2-loc / H6** | locality | Tabayyun-anchored vs best robust baseline | clean Recall_L2 ≥ best + 0.20, **and** worst-attack amnesia ≤ best − 0.20, **and** worst-attack ASR ≤ best baseline's |
| H3 | both | certificate | 0 violations over all Tabayyun runs |
| A2 | locality | corroboration gate needed | Tabayyun-noCorr worst-case (ASR or amnesia) ≥ Tabayyun-anchored + 0.20 |
| A6 | locality | key-anchored basis needed | Tabayyun-identity clean Recall_L2 ≤ Tabayyun-anchored − 0.20 |
| H1 | E01 | persistence | Lemma 1 exact on orthogonal streams; Lemma 1′ log-log r ≥ 0.95 on random streams (linear). *Confirmed Day 2–3; MLP uses α_eff = 2α (depth compounding), reported as exploratory.* |
| H5 | E01 | necessity | rayn ASR ≥ 0.9 under every per-write filter where single-shot ASR = 0. *Confirmed Day 3 (20/20 runs per filter).* |
| H4 | Day 24 | negative regime | Dirichlet(0.1) / unique facts: high-risk recall drops > 10 points (expected failure) |

Primary contrasts for Holm correction: P1 and P2-loc.

## Pre-declared negative regimes (reported whatever happens)
Dirichlet(0.1) heterogeneity; client-unique facts and facts with support < k (never promoted, by design); correlated upstream feed; Sybils beyond f.

---

## Amendment 1 (2026-11-10, after Day 4 confirmatory runs; tagged `prereg-v1-amend1`)
Every Day-4 result is reported as scored under v1 **and** under this amendment. Nothing from v1 is withdrawn.

1. **Floating-point tolerance (1e−9) in threshold comparisons.** Two Day-4 locality cells failed only because a bootstrap bound was −0.19999999999999996 against "≤ −0.20". Without the tolerance, locality P2 is 5/6 and A2 is 5/6; with it, both are 6/6. Both numbers are reported.
2. **Secondary metric `installed`** (‖read − v*‖ < 0.5‖v*‖). The registered ASR compares the read with v* and v_true. When the read is near zero (the fact was never promoted), that comparison is a coin toss and counts amnesia as a "flip". `installed` measures whether the attacker's value is actually in memory. The primary ASR is unchanged.
3. **New method variant `tabayyun_anchored2`** (a key is anchored once f+1 distinct sources wrote it, i.e. at least one honest source; promotion still needs k corroborators). Reason: Day 4 showed that the v2.1 anchoring rule (k writers inside a 5-round window) is an extra bottleneck not modelled by Thm 3(b). It is tested **confirmatorily on fresh seeds 110–119** with the same criteria (P1, P2-dense, P2-loc, A2, A6, H3).
4. **New hypothesis H7 (theory predicts the cost of security).** Clean Recall_L2 of the method equals the Thm 3(b) promotion probability P(Bin(m, 1−(1−ρ)^{R−1}) ≥ k), averaged over fact supports m, within ±0.10 in every cell. On Day-4 data (registered post hoc, so exploratory) it holds 8/9 dense and 5/6 locality. On the anchored2 probe (seeds 0–5) it holds 8/8 within 0.07. **Confirmatory on seeds 110–119.**
5. **Fairness ablation A11**: the robust baselines also get the key-anchored basis (`agg_*_anc`). **A3**: v1 additive promotion (`tabayyun_anchored2_add`). Both are reported, with no success threshold (descriptive).

---

## Amendment 2 (2026-11-10, before Day 6 runs; tag `prereg-v1-amend2`)
Day 5 confirmed Amendment 1: dense P1 9/9, P2 7/9, H7 9/9; locality all 6/6; 0/4500 violations. **The comparison set is now made stricter.**
1. **Extended baseline set for "best robust baseline" in P2:** mean, TrMean, median, Krum, GeoMed, CClip, median/Krum with the anchored basis, **SMSR-style** randomised ablation + voting (2606.12703), **RobustRAG-style** isolate-then-aggregate, **TMA-NM-style** corroboration-gated elevation (2606.24322: ≥ k distinct writers, plain mean), **Miras-Yaad** (Huber attentional bias), and Titans + per-write clip. Each criterion uses the best baseline *for that criterion*.
2. Same criteria (P1, P2-dense, P2-loc, H3, H7) and same method (`tabayyun_anchored2`), **fresh seeds 120–129**.
3. Ports are faithful to each paper's defining mechanism, adapted to parametric memory, and receive the same omniscient adversary (white-box control of their own sources' state). The adaptation choices are listed in `src/tabayyun/baselines.py`. A reviewer may dispute a port; we will release them for scrutiny.

---

## Amendment 3 (2026-11-10, after Day 6; prospective only)
Day 6 showed that "best baseline per criterion" is unsatisfiable when the set mixes undefended memories (full recall, ASR 1.0) with majority aggregators (ASR 0, recall 0). This was a flaw in Amendment 2, reported as a fail. **From Tier B onward, and for the final Tier-A rerun (Day 27, seeds 300+):**
1. **Primary security metric: `installed`** (attacker value present: ‖read − v*‖ < 0.5‖v*‖; for token models: the decoded token equals the attacker's token). Flip-ASR is secondary.
2. **Primary comparison (per cell):**
   - (C1) Pareto non-domination over (worst installed, worst amnesia, clean recall).
   - (C2) recall ≥ best *secure* baseline (worst installed ≤ 0.1) + 0.20, in the locality / local-knowledge regime.
   - (C3) worst installed ≤ best *useful* baseline (recall within 0.05 of the method) − 0.50.
   - (P1) installed ≤ 0.5 × Titans installed.
3. H3 (0 violations) and H7 (Thm 3(b) predicts recall ±0.10) unchanged.

---

## Amendment 4 (2026-11-10, Tier B confirmatory; registered BEFORE trials 200–299 are run)
**Design (from exploratory trials 0–99 and clean-only tuning, `results/e04/tuning_clean.json`):**
- Models: MemLM d128/dk64 and d256/dk128 (released in `checkpoints/`), 16 facts per trial.
- Cells: dense (n=10, f ∈ {1,2}) and locality (n=20, f ∈ {2,3}), ρ ∈ {0.4, 0.7}, **observation rounds R ∈ {1, 3}** (R = 3 added for realism; untuned). Attacks: assert, assert ×4, edge-of-range (white-box), unbounded (white-box).
- **Primary method `tabayyun`**: ν_rel = none, γ = 0.3, chosen by the pre-set clean-only objective (recall + certified-and-correct).
- **Secondary method `tabayyun_strict`**: ν_rel = 0.3, γ = 0.3 (strict k-witness semantics; see the Day 8 diagnosis of cross-talk corroboration).
- Baselines: Titans (ungated, concatenated), mean / TrMean / median / Krum / GeoMed of the per-source memories, RobustRAG-style, SMSR-style, TMA-NM-style.
- **Trials 200–299** (fresh).

**Criteria: Amendment 3, unchanged** (P1, C1, C2 in locality, C3) for both methods, plus H3 (0 certificate and 0 certified-token violations) and **H7 for `tabayyun_strict`** (|clean recall − Thm 3(b) prediction| ≤ 0.10, prediction conditioned on ≥ 1 observer).

**Expectations written down in advance (from 8 exploratory trials):** C3 (installed ≤ best useful baseline − 0.50) is expected to **fail** in most cells, because in token space the baselines install less (≈ 0.1–0.5) than in Tier A, so a 0.50 margin is unreachable. Amendment 3 stays as written and the failure will be reported. The primary method's H7 is expected to fail where cross-talk inflates recall above Thm 3(b) (it is reported but is not a criterion for the primary method).

---

## Amendment 5 (2026-11-11, adaptive gradient attacker; registered BEFORE the confirmatory run)
Only a 2-trial smoke run (trials 0–1, 30 steps) was seen, for debugging. It showed PGD installing at 0.25–1.00 against TrMean / median / GeoMed and at 0.25 against primary Tabayyun in one cell.
- **Attacker** (`experiments/e05_adaptive.py`): white-box, unbounded control of the f adversarial sources' memory states; Adam on the CW token loss through each defence's exact forward pass, with a BPDA cone hinge for the discrete gate; 3 restarts (edge-of-range / target-scaled / random); 100 steps; lr = 0.05 × honest scale.
- **Cells**: the 16 Tier-B cells of Amendment 4. **Trials 300–329** (fresh). Models: d128 and d256.
- **Criteria**
  - **H8a**: 0 certificate and 0 certified-token violations under PGD, for both Tabayyun variants.
  - **H8b**: `tabayyun_strict` PGD-installed ≤ 0.10 in every cell. (The primary variant is reported against the same threshold, descriptively.)
  - **H8c**: per cell, `tabayyun_strict` PGD-installed ≤ min(TrMean, median, GeoMed) PGD-installed + 0.05.
  - Krum and RobustRAG are reported with their Tier-B clean recall (secure but low-recall).

---

## Amendment 6 (2026-11-11, federation; registered BEFORE the confirmatory run)
Only a 2-trial smoke run (trials 0–1, 4 rounds) was seen.
- **Setup** (`experiments/e06_federated.py`): N = 12 clients × n = 7 sources (f = 1 adversarial source inside every client), 16 facts, 8 rounds. Byzantine clients f_c ∈ {1, 2} (first clients) send crafted rows **and** under-report their local bounds. Regimes: dense (every honest client relevant) and locality (each fact relevant at m ~ U[2f_c+1, N−f_c] honest clients), ρ ∈ {0.4, 0.7}. Attacks: none, local (edge-of-range inside clients + targeted writes), server_edge, server_unbounded, both.
- **Defences**: fed_tabayyun (local Tabayyun + server gate, Thm 5 certificate); local Tabayyun + server {mean, TrMean, median, Krum, GeoMed}; FedAvg over local ungated Titans (FedNL-style substrate).
- **Trials 400–419** (fresh).
- **Criteria**
  - **H9a**: 0 violations of the Theorem 5 certificate.
  - **H9b**: fed_tabayyun worst-case installed ≤ 0.05 **and** worst-case flip ≤ 0.10 in every cell.
  - **H9c**: fed_tabayyun is Pareto-non-dominated (worst flip, worst amnesia, −clean Recall_L2) by every federated baseline in every cell.
  - **H9d** (necessity at federation scale): FedAvg-Titans worst-case flip ≥ 0.5 in every cell.

---

## Amendment 7 (2026-11-12, scaling replication + ablations; registered BEFORE either run)
**(a) Scaling replication of Amendment 4 on MemLM d512/dk256** (released checkpoint, trained Day 10). Only the `--smoke` pipeline check (exploratory trials, outside 200–299) was run on it, to confirm the checkpoint loads. Same cells, attacks, methods (ν_rel, γ unchanged, no re-tuning), baselines and **trials 200–299** as Amendment 4. Criteria are **Amendment 4, unchanged**, scored by `scripts/analyze_e04.py`. Expectations stated in advance: C3 fails (as at d128/d256), and C2 holds only partly.
**(b) E07 component ablations** (`experiments/e07_ablations.py`, seeds 500–509, descriptive, no pass/fail). Only a 1-seed smoke run was seen. Predictions written down now:
- **A4** (momentum η_P > 0 in M^P): the momentum-free certificate is **not** valid. Violations are expected, with drift/cert > 1 that grows with η_P, as Thm 2′'s convexity argument predicts.
- **A5**: lower ν or γ trades recall for security only weakly. Removing the cone test keeps the certificate valid (Prop. R does not use it) but raises installed under edge-of-range. k + 2 lowers locality recall (Thm 3b). Every A5 variant keeps 0 violations.
- **A10**: reset or decay of the quarantine after promotion lowers recall in the locality regime (facts must be re-observed). Certificates stay valid.

---

## Amendment 8 (2026-11-12, Tier C: O-RAN slicing control on real ColO-RAN traces; registered BEFORE the confirmatory run)
**What was seen before registration** (exploratory only):
- clean-only tuning of the shared write step θ on trials 0–4 (`results/e08/tuning_clean.json`; θ = 0.05 is selected and used for **every** defence; it is the edge of the grid);
- a 2-trial smoke run of all attacks (trials 0–1).

**Design change disclosed before the run.** On real KPIs, the top configurations are near-ties (top-1 vs top-2 utility gap ≈ 0.005 in z units), so the exact-action form of Thm 3(a) certifies 0% of decisions. **Thm 3(a′)** (certified ε-optimality, `certify.certified_regret`, fuzz-tested) bounds the regret of the chosen action under the honest reference. The proposal's "certified-action fraction" is therefore measured as the fraction of decisions certified **10%-optimal** (ε̂ ≤ 0.1 × utility range). The exact fraction is also reported.

**Setup** (`experiments/e08_oran.py`):
- One xApp per base station (7 BSs, BS = trial mod 7) chooses among 84 actions (3 schedulers × 28 slice-PRB splits) by a linear SLA utility (eMBB throughput − URLLC buffer − 0.5 MTC buffer).
- Honest reports are real ColO-RAN measurement windows. Compromised reporters spoof KPIs so that the worst action looks best.
- Cells: dense (n = 10, f ∈ {1, 2}) and locality (n = 20, f ∈ {2, 3}), ρ ∈ {0.4, 0.7}, 10 control epochs.
- Attacks: spoof (targeted), rayn, edge-of-range, unbounded.
- Baselines: Titans, Titans+clip, Miras-Yaad, mean / TrMean / median / GeoMed / Krum, median+anchored, RobustRAG, SMSR, TMA-NM.

**Trials 600–619** (fresh).

**Criteria (8 cells):**
- **H10a**: 0 drift-certificate and 0 ε-certificate violations.
- **H10b**: Tabayyun worst-case final-epoch hijack ≤ 0.05 in every cell.
- **H10c**: Tabayyun is not Pareto-dominated on (worst hijack, worst regret, clean regret) by any baseline, in every cell.
- **H10d**: fraction of clean decisions certified 10%-optimal ≥ 0.60, averaged over cells (proposal H3, O-RAN).
- **H10e** (necessity): Titans worst-case hijack ≥ 0.5 in every cell.

**Expectations stated now:** `hijack_any` (any epoch, transient) is reported descriptively and may be > 0 for Tabayyun. The exact-action certified fraction is expected to be ≈ 0. In the 2-trial smoke, **H10c failed in 2/8 dense cells**: robust aggregators matched Tabayyun on security with ≈ 0.1 lower clean regret, a difference between near-tie actions. H10c is kept as written, and a fail there is possible.

---

## Amendment 9 (2026-11-14, limits and failure modes; registered BEFORE the run)
These are the negative regimes pre-declared in v1 ("reported whatever happens"). Only a 1-seed smoke run (seed 0, ρ = 0.7) of `experiments/e09_limits.py` was seen. **Seeds 700–709** (fresh), ρ ∈ {0.4, 0.7}, Tier A, n = 10.

- **H4 (expected failure, v1 criterion unchanged):** with client-unique facts (25% and 50% of facts known to one honest source) and with Dirichlet(0.1) observation, Tabayyun's high-risk Recall_L2 over all facts drops by **> 10 points** against the unique = 0 control.
  - H4-i: high-risk recall of unique facts ≤ 0.05 (never promoted, by design).
  - H4-ii: the owner's low-risk read of its own unique facts ≥ 0.90.
  - H4-iii: security is unaffected: installed ≤ 0.05 and 0 certificate violations in every H4 cell.
- **SYB (Sybils beyond f; gate assumes f = 2, world has f_act ∈ {1..5}):**
  - S-i: for f_act ≤ 2, installed = 0 and 0 violations.
  - S-ii: for f_act > 2, certificate violations are **expected** (the certificate is not claimed outside the threat model).
  - S-iii: at f_act = 5 = 2f+1, worst flip ≥ 0.5 (the adversary alone reaches k witnesses).
  - The breakdown curve of the baselines is reported next to it, descriptively.
- **UPS (correlated upstream feed: f = 2 adversaries + c ∈ {0..3} honest sources relaying a poisoned feed):** descriptive, no criterion. Stated in advance: the certificate bounds drift from the honest-source consensus, not from the truth, so it can stay "sound" while the read is wrong.
- **LIV (liveness under MedianShift):** in every cell, Recall_L2 under MedianShift ≥ clean recall − 0.05, and 0 violations.

---

## Amendment 10 (2026-11-14, Tier D: test-time memory on frozen pretrained LMs, CounterFact; registered BEFORE the confirmatory runs)
**Why.** Tiers A–C use synthetic memories, small trained memory LMs and a linear controller. Tier D tests the same defence on **pretrained language models** with a real knowledge dataset.

**Substrate** (`src/tabayyun/models/lm_retrofit.py`, `experiments/e10_lm_retrofit.py`):
- The LM is frozen. A Titans-style linear memory is added at the unembedding input: logits = W_U (x_out + gain · M k).
- Key k: PCA-whitened hidden state of the last prompt token (d_k = 256). The whitening map is fitted on 2,000 disjoint calibration prompts.
- Value: the unit unembedding direction of the asserted next token.
- The logits are exactly linear in the read, so Thm 3(a) applies exactly with W = gain · W_U.

**Data.** CounterFact (Meng et al., 2022), first 4,000 records (`data/counterfact/counterfact_4k.json`).
- Records 0–1,999 are calibration; records 2,000–3,999 are the trial pool.
- The stored knowledge is the record's **new** object, which the frozen LM does not know (base accuracy 0.2–0.4% on the two small models tried), so recall measures the memory.
- The attacker asserts another record's object for two target facts.

**Protocol.** Cells, attacks, defences, metrics and code path are those of Amendment 4 (`e04.apply_defences`, refactored with bit-identical smoke output). 16 facts per trial, R ∈ {1, 3}, both Tabayyun variants with γ = 0.3 and ν_rel as in Amendment 4 (no re-tuning of the gate).

**Models for the confirmatory run:** `gpt2-xl` (1.5 B) and `Qwen/Qwen2.5-1.5B`. **Trials 800–899** (fresh).

**Hyper-parameters chosen by a fixed clean-only rule, run on the confirmatory models before any attacked trial** (`--tune`: attack = none, exploratory trials 0–4):
- grid: key layer ∈ {L/3, L/2, 2L/3, L} × read boost ∈ {25, 50, 100} logits;
- objective: mean over both Tabayyun variants and all cells of clean recall + certified-and-correct fraction;
- the selected values are written to `results/e10/tuning_<model>.json` and used for every defence.

**What was seen before registration** (small models only, exploratory):
- 3-trial smoke runs and the clean-only tuning on `gpt2` (124 M) and `Qwen/Qwen2.5-0.5B` (`results/e10/tuning_*.json`).
- `gpt2` smoke: 0 violations; Titans installed 0.50; robust aggregators installed 0.00; Tabayyun certified 0.79–0.96 of tokens.
- Tuning: the objective is flat in layer and in boost ≥ 50 (`gpt2` 1.33–1.34, Qwen-0.5B 1.24–1.28). Both select boost = 100, the edge of the grid; the grid is kept as registered.
- After the switch to logit-boost units, Qwen-0.5B clean recall equals `gpt2`'s (0.72).
- First Qwen smoke at a GPT-2-sized gain: recall was low for every method, because Qwen's unembedding rows are 8× smaller. The gain is therefore expressed as a logit boost (model-independent).

**Criteria: Amendment 4, unchanged, per model** (`scripts/analyze_e04.py`): P1, C1, C2 (locality), C3, H3 (0 certificate and 0 certified-token violations), and H7 for `tabayyun_strict`.

**Expectations stated now:**
- **C3 is expected to fail**, as in Tier B, and **C2 is expected to fail in most cells** (in token space robust aggregators keep high recall).
- The claims under test are **H3, P1, C1 and strict H7**.
- C1 may fail in dense cells, as it did at d512 and in Tier C dense.

---

## Amendment 11 (2026-11-15, adaptive gradient attacker on Tier D; registered BEFORE the confirmatory runs)
**Why.** Amendment 10 used static attacks, under which TrMean / median / GeoMed look secure in dense cells. On Tier B the adaptive attacker installed at 1.00 against TrMean and GeoMed (Amendment 5). The same attacker is now run on the pretrained-LM substrate.

**Setup** (`experiments/e11_lm_adaptive.py`):
- attacker code: `e05.attack_all` (refactored out of E05 with bit-identical smoke output); 100 Adam steps, 3 restarts, CW token loss, unbounded adversarial rows;
- substrate, facts, cells and clean-tuned (layer, boost): those of Amendment 10 (`results/e10/tuning_<model>.json`, already fixed);
- defences: both Tabayyun variants, TrMean, median, GeoMed, Krum, RobustRAG;
- models: `gpt2-xl` and `Qwen/Qwen2.5-1.5B`; **trials 900–919** (fresh), 16 cells.

**What was seen before registration:** a 2-trial, 30-step smoke run on `gpt2` (124 M), 4 cells: PGD installed 0.25 against TrMean, median or GeoMed in 3 of 4 cells, 0.00 against both Tabayyun variants, 0 violations.

**Criteria: Amendment 5, unchanged, per model** (`scripts/analyze_e05.py`):
- **H8a**: 0 certificate and 0 certified-token violations under PGD, both Tabayyun variants.
- **H8b**: `tabayyun_strict` PGD-installed ≤ 0.10 in every cell (primary reported against the same threshold, descriptively).
- **H8c**: per cell, `tabayyun_strict` PGD-installed ≤ min(TrMean, median, GeoMed) PGD-installed + 0.05.

**Expectation stated now:** the primary variant may exceed 0.10 in locality R = 1 cells, as on Tier B.

---

## Amendment 12 (2026-11-15, union adaptive attacker on Tier D; registered BEFORE the confirmatory runs)
**Why.** The Amendment 11 run passed, but the gradient attacker installed **0.00** against primary Tabayyun in every cell, while the plain write attack of Amendment 10 installed 0.09–0.14 in the hardest cell (locality, f = 3, ρ = 0.4, R = 1). An adaptive attack that is weaker than a static one is a failed attack, not evidence of security. Two causes were found on `gpt2` (exploratory trials 0–59):
- the Amendment 11 adversary set its rows directly but did not write camouflage keys, so fewer directions were anchored;
- the optimiser never started from, or kept, the state a plain write attack produces.

**Union attacker** (`e11_lm_adaptive.py --union`, `e05.attack_defence` with the new options; E05 defaults unchanged and bit-identical on smoke):
- extra starting points: the adversary's **real quarantines** under `assert` and `assert_x4`, with its camouflage keys counted as written keys;
- every start is also kept un-optimised;
- for Tabayyun the optimisation runs with **and** without the cone hinge;
- the final choice per trial maximises the number of targets where the attacker's token wins, then the CW loss.
So the union attacker is at least as strong as the static write attacks on the same trial, by construction.

**Setup.** Models `gpt2-xl` and `Qwen/Qwen2.5-1.5B`, clean-tuned (layer, boost) of Amendment 10, 16 cells, 100 steps. Defences: both Tabayyun variants and TrMean (reference). **Trials 920–939** (fresh).

**What was seen before registration** (`gpt2`, 124 M, exploratory):
- static replica vs the Amendment 10 code on trials 0–59: primary 0.108 vs 0.125 (locality f = 3) and 0.042 vs 0.067 (dense f = 2); strict 0.008 vs 0.008; TrMean 0.158 vs 0.192;
- a 4-trial, 30-step union smoke on two cells: Tabayyun 0.00, TrMean 0.12 and 1.00, 0 violations.

**Criteria, per model** (`scripts/analyze_e11u.py`):
- **H12a**: 0 certificate and 0 certified-token violations.
- **H12b**: `tabayyun_strict` installed ≤ 0.10 in every cell under the union attacker.
- **H12c** (attack sanity): for each of the three defences, the union attacker's installed rate averaged over cells ≥ the static write attacks' rate on the same trials − 0.02.

**Expectation stated now:** primary Tabayyun exceeds 0.10 in the locality f = 3, R = 1 cells. It is reported descriptively and is not a criterion. The paper will report, for every defence, the **worst over all attacks run** (static, PGD, union).

---

## Amendment 13 (2026-11-15, union adaptive attacker on Tier B; registered BEFORE the confirmatory runs)
**Why.** Amendment 5 (Tier B, MemLM) used the same optimiser that Amendment 12 showed to be no stronger than a plain write attack against the Tabayyun gate. Its Tabayyun numbers (strict ≤ 0.083, primary ≤ 0.42) are therefore re-tested with the union attacker. Tier B is the harder case for the gate, because honest quarantines differ from each other there (cross-talk), which leaves the attacker more room than on Tier D.

**Setup** (`experiments/e05_adaptive.py --union`; default path unchanged and bit-identical on smoke):
- the union attacker of Amendment 12: starts from the adversary's real quarantines under `assert` and `assert_x4` (camouflage keys counted as written keys), every start kept, optimisation with and without the cone hinge, selection by number of targets won;
- models: MemLM d128, d256 and d512 (released checkpoints); 16 cells; 100 steps;
- defences: both Tabayyun variants and TrMean (reference);
- **trials 330–359** (fresh).

**What was seen before registration:** a 2-trial, 30-step union smoke on d128 (4 cells): Tabayyun 0.00 in all 4, TrMean 0.25 / 0.00 / 1.00 / 1.00, 0 violations.

**Criteria, per model: those of Amendment 12** (`scripts/analyze_e11u.py`): H12a (0 violations), H12b (`tabayyun_strict` installed ≤ 0.10 in every cell), H12c (union ≥ static − 0.02 on average, for each defence).

**Expectations stated now:** the primary variant exceeds 0.10 in locality R = 1 cells (it reached 0.32–0.42 under Amendment 5). **H12b may fail on Tier B**; if it does, it is reported as a fail and the strict variant's secure-configuration claim is limited accordingly.

---

## Amendment 14 (2026-11-15, witness complexity, Theorem 6, and the witness-optimal variant; registered BEFORE the confirmatory run)
**Why.** Across Tiers A–D, Tabayyun's advantage over robust aggregators appears only when facts are known to few sources. Theorem 6 (`paper/proofs.md`) states this as a result: the number of honest witnesses a rule needs is f+1 for bounded error, 2f+1 for hull validity, and about n/2 for majority aggregation. It also yields a **new variant, Tabayyun-W** (`GateConfig(rule="agree")`, defence `tabayyun_w`): promote when f+1 active sources agree within a tolerance ε, with the range certificate Prop. R′.

**Experiment** (`experiments/e12_witness.py`): Tier A; (n, f) ∈ {(10,2), (20,2), (40,2), (80,2), (40,1), (40,3)}; every fact is known to exactly m honest sources, m on a grid from 1 to n − f; witnesses re-assert every round (ρ = 1), 6 rounds. Clean recall per m; attacks on one fact per m: targeted writes, edge-of-range, unbounded, and `agree_edge` (omniscient attack on the agreement gate that solves for the largest admissible shift). Defences: Titans, mean, TrMean, median, GeoMed, RobustRAG, TMA-NM, Tabayyun (cone, k = 2f+1), Tabayyun-W (agree, k = f+1). **Seeds 1000–1009** (fresh).

**Definitions fixed now.** Witness threshold m\* = the smallest grid m from which clean recall ≥ 0.5 at that m and every larger m. Flip = a non-negligible read (‖y‖ ≥ 0.1‖v‖) that is closer to v\* than to v (for an unserved fact the read is ≈ 0 and "closer" is a floating-point coin toss; found in the smoke run and fixed before registration).

**ε chosen by a fixed clean-only rule** (`scripts/tune_e12_clean.py`, seeds 0–4, attack = none): the smallest ε on {0.1, 0.15, 0.2, 0.3, 0.4, 0.5} whose clean recall at m ≥ f+1 is within 0.02 of the best. Result: **ε = 0.2** (`results/e12/tuning_clean.json`).

**What was seen before registration:** the clean tuning, and a 2-seed smoke on (10,2) and (40,2): thresholds 5 / 3 / ≈ n/2, Tabayyun variants 0 flips and 0 violations, TMA-NM and Titans flipped.

**Criteria** (`scripts/analyze_e12.py`):
- **W1**: Tabayyun (cone) has m\* = 2f+1 in all 6 configurations.
- **W2**: Tabayyun-W has m\* = f+1 in all 6 configurations.
- **W3**: mean, TrMean, median, GeoMed and RobustRAG have m\* ≥ 0.4·n in every configuration with n ≥ 20.
- **W4**: both Tabayyun variants: worst installed ≤ 0.05 and worst flip ≤ 0.05 over every (m, attack); amnesia ≤ 0.10 at every served m; 0 certificate violations.
- **W5**: Titans and TMA-NM have worst flip ≥ 0.5 in every configuration.

**If W2 and W4 hold**, Tabayyun-W is carried to Tiers B and D under new amendments (same trials protocol, adaptive attacks included) before any claim about it is made.

---

## Amendment 15 (2026-11-16, Tabayyun-W and certify-or-abstain serving on Tier B under the union attacker; registered BEFORE the confirmatory run)
**Why.** Amendment 13 failed (H12b: strict installed 0.15 at d256 and 0.283 at d512 in locality f = 3, ρ = 0.4, R = 1). The leak path: when a target fact is not promoted, its read is near zero, so a bounded push through cross-talk from other promoted directions decides the argmax. Two remedies follow from the theory, and neither has been confirmed yet:
1. **Tabayyun-W** (Thm 6b): f+1 agreeing witnesses promote, so fewer target facts are left unpromoted and open to the leak.
2. **Certify-or-abstain serving:** emit a token only if Thm 3(a) certifies it, otherwise abstain. Under the certificate, a served token equals the token of the honest-corroborator reference.

**ε of Tabayyun-W on Tier B** was chosen by a fixed clean-only rule (`scripts/tune_e13_clean.py`: attack = none, exploratory trials 0–4, d128 and d256, objective clean recall + certified-and-correct, smallest ε within 0.02 of the best): **ε = 0.4** (`results/e05/tuning_w_clean.json`). The gate uses ν_rel = 0.3, as the strict variant does.

**Attacker.** The union attacker of Amendment 13, plus two moves specific to Tabayyun-W:
- an `AgreeEdge` start (just inside the agreement ball, towards the target);
- a BPDA hinge that keeps adversarial rows within 0.9 r of the honest mean.

**What was seen before registration** (exploratory trials 0–9, 60 steps):
- Tabayyun-W installed 0.000 in every cell at d128 and d512, with the gate-specific moves included;
- strict 0.034 / 0.044 mean, worst cell 0.20 / 0.35;
- amnesia: W 0.13–0.14, strict 0.35;
- certified-and-correct targets under attack: W 0.54–0.58, strict 0.006–0.066;
- 0 violations.
- **Control:** at ε = 2.0 the same attacker installs 0.55 against Tabayyun-W (locality f = 3, ρ = 0.4, R = 1, trials 0–9), so the attacker is capable.

**Confirmatory run:** MemLM d128, d256, d512; 16 cells; 100 steps; defences `tabayyun`, `tabayyun_strict`, `tabayyun_w`, `agg_trmean`; **trials 360–389** (fresh). Plus a control run: d128, ε = 2.0, locality f = 3, ρ = 0.4, R = 1, same trials.

**Criteria** (`scripts/analyze_e15.py`, all three models):
- **H15a**: 0 certificate and 0 certified-token violations (strict and W).
- **H15b**: Tabayyun-W union-installed ≤ 0.05 in every cell.
- **H15c** (certify-or-abstain): served-installed ≤ 0.02 in every cell, for strict and W.
- **H15d**: Tabayyun-W mean amnesia under attack ≤ strict − 0.10.
- **H15e** (attack sanity): union ≥ static − 0.02 on average, for strict, W and TrMean.
- **Control**: union-installed ≥ 0.20 against W at ε = 2.0.

**Expectation stated now:** Tabayyun-W has larger worst-case drift per promoted fact than strict (shift up to 2r when it is promoted on fewer than 2f+1 witnesses). Its clean recall against the other variants on Tiers B and D is tested separately (Day 21), not here.
