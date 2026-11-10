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
