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
