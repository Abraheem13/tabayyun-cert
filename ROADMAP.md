# Roadmap: 30 days of experiments, then the paper

**Working protocol (every day)**
1. `git pull`, then run the day's command(s) on the RTX 5080 (`scripts/run_day.sh <day>`).
2. Commit `results/registry.jsonl` and `results/<exp>/*.json` and push. Never commit model checkpoints.
3. Claude analyses the numbers, appends to `EXPERIMENT_LOG.md`, fixes or extends code, and pushes. Ablations go to `ABLATIONS.md`, design changes to `CHANGELOG.md`.
4. A day counts as done only when the unit tests pass (`pytest -q`) and the log entry exists.

**Go / no-go gates.** Days 7, 14, 21 and 30. If a gate fails, we fix or re-scope before moving on. We never move on with a broken claim.

GPU estimates are for one RTX 5080 (16 GB, bf16). "CPU" means it also runs on a laptop.

---

## Phase 0 (Day 0, today). Setup ✅
- Repo, core library (memory, gate, certificates, attacks, baselines, testbed), 50 passing tests.
- Proposal v2 (12 fixes), novelty audit v2, pre-registration draft, this roadmap.

## Week 1: theory-grade testbed (Tier A)
| Day | Work | Output | Compute |
|---|---|---|---|
| 1 | Environment on the RTX 5080 (PyTorch cu128, sm_120 check), full test suite, smoke runs | `env.txt`, green tests | 10 min |
| 2 | **E01** full persistence grid (linear + MLP, 7 η × 6 α × 5 seeds); orthogonal vs random streams | H1 figure | 1 h GPU |
| 3 | **E01b** rayn vs single-shot vs per-write norm filters (clip G sweep); write the Prop. 0 proof | H5 table, proof draft | 30 min |
| 4 | **E02** full grid: 10 defences × 6 attacks × 3 f/n × 10 seeds; regime map over ρ ∈ {0.1..0.9} | H2 (Tier A), H6 | 2 h |
| 5 | Certificate study: soundness over all runs; tightness ratio; EMA vs additive (F3); write Thm 2′ proof | H3 (Tier A) | 1 h |
| 6 | Port the new baselines: SMSR (ablation + voting), TMA-NM corroboration, RobustRAG, Miras-Yaad (Huber) | baselines v2 | CPU |
| 7 | **Gate 1.** Freeze the pre-registration (tag `prereg-v1`). Go if 0 certificate violations, H6 holds in the sparse regime, and Tabayyun ≥ best baseline on adaptive attacks | review note | — |

## Week 2: learned memories (Tier B)
| Day | Work | Output | Compute |
|---|---|---|---|
| 8 | MQAR data generator (Zoology protocol); Titans-style LMM (MAC-lite) training code; bf16 | `models/`, `data/mqar.py` | — |
| 9 | Train LMM-25M and LMM-60M on MQAR plus a small language mix; clean recall baseline | checkpoints, recall table | 6–8 h |
| 10 | Tabayyun inside the trained model (per-source quarantine at inference, deep quarantine plus linear tier); clean-recall parity (≤ 1 pt) | parity table | 2 h |
| 11 | CounterFact injection streams (multi-source); memory retrofit on frozen SmolLM2-360M / Qwen2.5-0.5B | retrofit v0 | 6 h |
| 12 | BABILong qa1–qa5 at 4k/8k/16k/32k with the retrofit, clean and attacked | BABILong table | 4 h |
| 13 | **Adaptive attacker v2**: differentiable relaxation of the gate (soft median, sigmoid cone, soft trimming), PGD on the attacker's quarantine; BPDA for the hard gate | strongest-attack numbers | 4 h |
| 14 | **Gate 2.** H2 on MQAR and CounterFact incl. the adaptive attacker. If H2 fails: diagnose (ν/γ, basis, τ) on the *clean validation* split only | review note | — |

## Week 3: federation and O-RAN (Tier C)
| Day | Work | Output | Compute |
|---|---|---|---|
| 15 | Federated substrate (N = 10–20 clients, FedNL-style 3-level); client-level gate; FL baselines (FedAvg, Krum, Multi-Krum, TrMean, Median, RFA, CClip, FLTrust) | `federated.py` | CPU/GPU |
| 16 | Federated attacks (f_c malicious clients, crafted Δ^P); two-level certificate composition (proof + test) | federated table | 2 h |
| 17 | ColO-RAN / COMMAG pipeline: KPI traces → sources (E2 streams, xApp reports, tickets); memory-augmented slicing/scheduling policy | O-RAN env | 2 h |
| 18 | O-RAN attacks: spoofed KPIs, poisoned tickets → action flips; certified-action fraction; KPI regret (throughput, latency, SLA) | H3 (O-RAN) | 3 h |
| 19 | TeleQnA copilot with the retrofit (optional, only if Days 17–18 are green) | copilot table | 3 h |
| 20 | Liveness: MedianShift and denial-of-promotion; Thm 3b delay law vs measured; (k, τ) operating curve; Prop. 5 | liveness figure | 1 h |
| 21 | **Gate 3.** Federation and O-RAN claims hold, or are re-scoped to Paper B | review note | — |

## Week 4: ablations, negative regimes, tightness, reproducibility
| Day | Work | Output | Compute |
|---|---|---|---|
| 22 | Ablations: no quarantine; no corroboration (TrMean only); momentum in M^P; additive vs EMA; sweeps over ν, γ, β, k, τ, α^Q/α^P | ablation table | 3 h |
| 23 | Ablations: basis (identity / keys-SVD / robust-SVD / basis-rotation attack); static vs learned trust; linear vs random-feature promoted tier | ablation table 2 | 2 h |
| 24 | **H4 negative regime**: Dirichlet(0.1), client-unique facts; report failures honestly | H4 figure | 2 h |
| 25 | Out-of-threat-model failure curves: correlated upstream feed, Sybils beyond f, compromised aggregator | failure-mode table | 2 h |
| 26 | Scaling: n sources, d, K, memory size; wall-clock and communication per tier | cost table | 2 h |
| 27 | **Final runs**: all primary contrasts re-run from a clean checkout with all seeds; bootstrap CIs; Holm | frozen numbers | 8–10 h |
| 28 | Tightness: Thm 4 lower-bound attack vs Prop. R certificate; write the Thm 4 proof (or downgrade it to an empirical claim) | tightness figure | 1 h |
| 29 | Reproducibility: `make paper-numbers` regenerates every table and figure from `registry.jsonl`; release benchmark v1 | artefact | — |
| 30 | **Gate 4 (final).** Results freeze (tag `results-v1`); claim-to-evidence map; paper outline | `paper/outline.md` | — |

## After Day 30
- Days 31–45: write the TIFS paper (`paper/`, IEEE Transactions template). Internal red-team review: one week of reviewer-style attacks on our own claims.
- Days 46–50: re-run the novelty search in IEEE Xplore, ACM DL and OpenReview; partner-group review; submit to **IEEE TIFS**.
- In parallel: Paper B (O-RAN instantiation) for IEEE JSAC / TMC.

## What "success" means (set before the results, so we don't fool ourselves)
- **Must have** (or no submission): 0 certificate violations; H2 P1 and P2 on Tier A and at least one Tier-B benchmark incl. the adaptive attacker; H5 necessity.
- **Should have**: H3 tightness ≤ 10×; O-RAN certified-action ≥ 60%; federated results.
- **Honest reporting**: H4 and the out-of-model failures appear in the paper whatever happens.
