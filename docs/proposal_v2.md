# Tabayyun-Cert, Proposal v2.0 (8 Nov 2026)

**Certified, corroboration-gated quarantine for poisoned test-time memory in federated, agentic AI-native networks**

PI: Raja Abraheem Rashid Ejaz (Brunel University London; TRL, IBA Karachi)
Primary target: **IEEE Transactions on Information Forensics and Security (TIFS)**, Q1.
Alternates: IEEE TDSC. Companion paper (O-RAN instantiation): IEEE JSAC / TMC / TCCN.

v1 is the PDF proposal. This document records **what changed and why**, then the corrected method, theory and plan. Every change was made because a reviewer at a Q1 security venue would otherwise reject or weaken the paper.

---

## 0. Executive summary of the revision

| # | Problem in v1 | Consequence | Fix in v2 |
|---|---|---|---|
| F1 | Median and trimmed mean (Eqs. 5, 7) run over **all n sources**. Honest sources that never observed a fact contribute zeros. | The median is about 0, so the cosine test (Eq. 6) is undefined. The trimmed mean shrinks every true fact towards 0. The honest spread Δ_H becomes about ‖v‖, so **Theorem 2 is vacuous in exactly the regime it is meant for.** | Median over the **active** set A_u. Trimmed mean over the **corroborators** C_u only. Requires k ≥ max(2β+1, f+1) and β ≥ f. Δ_H is now the spread among honest **corroborators**, which is small. |
| F2 | Δ_H is not observable at run time (it needs to know who is honest). | The certificate D_R **cannot be computed** by the operator, so the abstention rule (Alg. 1, line 5) cannot be implemented. | **Proposition R (runtime certificate)**: an order-statistic bound computable from the corroborators' values alone, sound against any adversary with ≤ f sources. |
| F3 | Additive promotion M^P ← (1−α^P)M^P + Δ^P. | The certificate grows like 1/α^P, which is loose for the slow tier (small α^P). Facts seen every round also inflate to v/α^P. | **EMA (convex) promotion**: M^P u ← (1−α^P) M^P u + α^P · value_u for promoted u. The error recursion is a convex combination, so the bound is **≤ √(d_v K)·Δ_H with no 1/α^P factor**. The v1 rule is kept as an ablation. |
| F4 | Theory for linear memory. The MLP case relies on local linearisation. | Reviewers will call the certificate heuristic for deep memories. | **Deep quarantine, linear certified tier.** The quarantine can be any deep memory, and the gate reads it in function space, D[s,u] = M_s^Q(u). M^P is linear and momentum-free by construction, so **Theorems 1–3 are exact for every quarantine architecture.** For capacity, the option is a linear tier on random features φ(k). |
| F5 | The key basis is computed from recent keys of all sources. | The adversary controls part of the keys, so it can rotate the basis. The per-direction certificate also needs a fixed basis. | The basis is identity, or a robust (trimmed) key covariance, **frozen after round 1**. This is stated in the threat model. |
| F6 | Action certificate m(q) > 2L‖q‖D_R. | Loose: 2L and ‖q‖·‖ΔM‖_F. | **Pairwise, per-query certificate**: z_a − z_j > ‖w_a − w_j‖·Σ_u \|⟨u,q⟩\|·E_u. This is tighter for queries aligned with few directions. |
| F7 | H1 half-life law ln2/(−ln max(η,1−α)). | Day-0 smoke runs show it ignores momentum whenever η < 1−α. The exact single-write curve Σ_j (1−α)^{T−j} η^j has a momentum-amplified peak and a longer tail. Benign interference shortens persistence below the lemma. | H1 is restated on the **exact curve** (Lemma 1, which is exact for orthogonal streams and an upper bound otherwise). Persistence is measured with a **counterfactual twin** (poisoned vs clean memory on identical streams). |
| F8 | The novelty audit misses five 2026 papers (Sec. 5). One of them (TMA-NM, arXiv 2606.24322) already proposes *Sybil-resistant corroboration-gated elevation* for agent memory. | "Corroboration gating" alone is **not novel**. | The contribution is repositioned around (i) parametric, superposed memory, where provenance is destroyed at write time, (ii) quantitative drift and action certificates, (iii) federation, and (iv) a **necessity + sufficiency + tightness** theory triad. |
| F9 | Compute plan: 340M pretraining on 4×L40S. | Infeasible on one RTX 5080 (16 GB). | Rescaled plan (Sec. 7). Synthetic and learned small memories (≤60M) are trained from scratch, and a memory retrofit is added to a frozen ≤0.5B LLM. Everything fits in 16 GB with bf16. |
| F10 | IoT intrusion datasets (CICIoT2023, N-BaIoT) are listed with no role in any hypothesis. | Reviewers flag unexplained datasets. | Moved to optional. Replaced by **ColO-RAN + COMMAG** KPI traces (wineslab) as the telemetry sources, plus TeleQnA for the copilot. |
| F11 | Baselines miss the new certified memory defences. | A Q1 reviewer will ask for them. | Added: SMSR (randomised ablation + voting), TMA-NM-style origin-bound corroboration, RobustRAG isolate-then-aggregate, Miras-Yaad (Huber attentional bias), geometric median / RFA, and centered clipping. |
| F13 (Day 3) | A fixed basis mixes facts, so per-direction corroboration is not per-fact corroboration. | Low utility when sources observe different facts (identity basis: Recall_L2 0.03–0.13 in the locality probe). | **Key-anchored, growing, frozen basis**: a key becomes a direction once ≥ k distinct sources wrote it. Directions never change, so the certificate stays valid. |
| F14 (Day 3) | H2 "≥ 25% better than the best robust baseline" fails where every source sees every fact (median and Krum also reach ASR 0). | The headline claim would have been false. | Two regimes pre-registered: **dense** (non-inferiority plus certificate) and **locality** (facts known to a minority; robust aggregators erase them). |
| F12 | The liveness / denial-of-promotion attack (adversary shifts the median so honest sources fail the cone test) is not modelled. | It is an obvious reviewer attack on the gate. | Added as adversary goal (e). We implement `MedianShift` and report a liveness result (Prop. 5). |

---

## 1. The gap, sharpened

Agent-memory defences in 2026 split into two groups:

* **Discrete stores** (RAG and vector memory). SMSR certifies with randomised ablation and voting. TMA-NM binds authority to a write-time origin and elevates by Sybil-resistant corroboration. Belief-memory work caps the influence of each provenance. All of them rely on the fact that **each stored item keeps its own provenance label**.
* **Parametric test-time memory** (Titans / Miras / ATLAS / Hope, and FedNL federating them). Here a write is a gradient step, so **every write superposes into the same weights and the provenance is destroyed at write time**. TMA-NM's own Theorem 2 shows that write-time origin binding is *necessary*. For parametric memory there is no item to bind.

**Our thesis.** For parametric memory, origin binding is realised by **source-separated quarantine before superposition**. Only corroborated content is superposed into the shared, action-driving tier. This can be done with closed-form, run-time-computable guarantees on (i) how far ≤ f sources can move the trusted memory and (ii) whether they can flip a network action, also under federation.

Nobody has (a) an attack-persistence law for gradient-written memory, (b) a necessity result for source separation in parametric memory, (c) a certified drift and action bound for it, or (d) a federated version. Searches on 8 Nov 2026 found none (Sec. 5).

---

## 2. Threat model (changes from v1 in **bold**)

The system, sources and federation are as in v1. The adversary controls ≤ f sources per client and ≤ f_c clients. Its knowledge is white-box with respect to the defence.

* **The adversary may place any state in the quarantines of its own sources.** This is the strongest write-level adversary, because it controls every write of its sources. It is implemented as a promotion-time hook.
* Goals: (a) targeted insertion, (b) amnesia, (c) action flipping, (d) degradation, **(e) denial of promotion (liveness)**.
* **Basis**: fixed, or frozen after round 1 from a robust key covariance. Basis-rotation attacks are evaluated as a failure mode.
* Out of scope (each measured as a failure mode): Sybil identities beyond f; correlated upstream feeds (ASI04); a compromised aggregation server; backbone jailbreaks.

---

## 3. Method v2

Notation follows Titans/Miras. Source s has a quarantine memory M_s^Q (linear or deep MLP), updated by the Titans rule with a fast forget gate α^Q, momentum η and per-write clip G. The promoted memory M^P ∈ R^{d_v×d_k} is **linear and momentum-free**.

**Read (risk-routed).** y = M^P q + λ·1[r(q) ≤ r₀]·M^Q_{s(q)}(q).

**Gate, every τ steps, for each u in a fixed orthonormal basis U = {u_1..u_K}:**

1. D[s,u] = M_s^Q(u) (function-space read; for linear memory this is M_s^Q u).
2. A_u = {s : ‖D[s,u]‖ ≥ ν} (active sources).
3. m_u = coordinate-wise median of D[s,u] over s ∈ A_u.
4. C_u = {s ∈ A_u : cos(D[s,u], m_u) ≥ γ} (corroborators).
5. Promote u iff |C_u| ≥ k, where **k ≥ max(2β+1, f+1) and β ≥ f**.
6. value_u = TrMean_β of D[s,u] over s ∈ C_u.

**EMA promotion:** M^P ← M^P + Σ_{u promoted} α^P (value_u − M^P u) uᵀ. Unpromoted directions decay by (1−ε^P).

**Runtime certificate (per direction and coordinate):** let c = |C_u|, and let L_i (U_i) be the mean of the c−f smallest (largest) corroborator values in coordinate i. Then

  e_{u,i} = max(value_{u,i} − L_i, U_i − value_{u,i}),
  E ← (1−α^P)E + α^P e on promoted directions, and E ← (1−ε^P)E on the rest.

**Abstain rule:** a high-risk query is served only if its action is certified (Thm 3a) under drift bound Σ_u |⟨u,q⟩| E_u. Otherwise it is routed to a safe fallback (NTC-style).

**Federation:** clients send their promoted values and corroborator counts per direction. The server applies the same gate across clients with k_c ≥ max(2β_c+1, f_c+1). The certificates compose: the client-level error enters the server-level honest spread.

---

## 4. Theory v2

The proof status of each result is tracked in `docs/theory_status.md`. A result is called a theorem in the paper only when its proof is written and checked by the unit tests in `tests/`.

* **Lemma 1 (persistence, exact).** With constant gates, a single write's contribution after T steps is θg·Σ_{j=0}^{T}(1−α)^{T−j}η^j. This is exact when later writes are orthogonal to k*, and an upper bound otherwise, since self-correction only shrinks it. *Checked to 1e-8 in `test_persistence_lemma_exact`.*
* **Proposition 0 (necessity, new).** For any single-tier gradient-written linear memory with constant gates and any per-write norm filter G > 0, one source can drive M(k*) to within ε of any v* using O(‖v*‖/(θG)·log(1/ε)) writes. So without source separation no norm-based filter bounds a minority's influence. *Proof: rayn construction. Empirical side: H5.*
* **Theorem 1 (Sybil-limited promotion).** A direction is promoted only if ≥ k−f honest sources corroborate it. With no honest component above ν along u, the adversary's influence along u is exactly 0. *Fuzz-tested.*
* **Lemma 2 (hull).** If c ≥ 2β+1 and β ≥ f, every coordinate of value_u lies in the coordinate-wise range of the honest corroborators. *Fuzz-tested against arbitrary rows.*
* **Theorem 2′ (a-priori drift, EMA rule).** Let M̃^P be the promoted memory under the same decisions with the honest-corroborator mean. Then ‖M^P_R − M̃^P_R‖_F ≤ √(d_v K)·Δ_H^C for every R, where Δ_H^C is the honest-corroborator spread. Unlike v1 there is no 1/α^P factor. The v1 additive rule gives √(d_v K)Δ_H(1−(1−α^P)^R)/α^P.
* **Proposition R (runtime certificate, new).** ‖M^P_R − M̃^P_R‖_F ≤ ‖E_R‖_F. For every query, ‖(M^P−M̃^P)q‖ ≤ ‖Σ_u|⟨u,q⟩|E_u‖. Both are computable without knowing who is honest. *Checked end-to-end under edge-of-range, unbounded and median-shift attackers.*
* **Theorem 3 (action invariance and promotion delay).** (a) Pairwise linear-head form: if z_a − z_j > ‖w_a−w_j‖·drift(q) for all j, the action equals the honest-reference action. (b) P(promoted by r) = P(Bin(n_H, 1−(1−ρ)^r) ≥ k). *(b) matches simulation within 0.015.*
* **Theorem 4 (tightness, to prove).** The edge-of-range adversary achieves drift Ω(f/(c−2β)·Δ_H^C) per direction. Together with Prop. R this shows the certificate is tight up to a factor depending only on (c, f, β). Empirical side: H3.
* **Proposition 5 (liveness, to prove).** If the honest active values along u lie in a ball B(c_u, r_u) with r_u/‖c_u‖ ≤ ρ(γ) and |A_u ∩ H| ≥ 2f+1, then every honest source passes the cone test. So a median-shift attacker can deny promotion only when honest support is thin. Empirical side: promotion-delay metric under `MedianShift`.

---

## 5. Novelty audit v2 (web searches, 8 Nov 2026; to be re-run in IEEE Xplore / ACM DL / OpenReview before submission)

| Closest work | ID | Memory type | Guarantee | Corroboration | Federation | What it lacks relative to Tabayyun-Cert |
|---|---|---|---|---|---|---|
| SMSR | 2606.12703 | RAG store | Certified ASR bound via randomised ablation + voting; loose for many sessions | No | No | Not parametric; no drift or action bound |
| TMA-NM (Louck) | 2606.24322 | Discrete long-term memory | Machine-checked soundness on a bounded model | **Yes, Sybil-resistant corroboration-gated elevation** | No | Needs per-item origin, which parametric memory destroys; binary safety, no quantitative drift |
| Belief-based memory | 2606.22030 | Belief store | Provenance-capped influence | Partial | No | Not parametric; no action certificate |
| MEMSAD | 2605.03482 | RAG agents | Detection (empirical) | No | No | Heuristic |
| Utility under attack | 2608.21230 | Agent memory | Empirical study | — | No | Study only |
| ISNAD | 2607.24117 | External claims | Routing (no proof) | Cross-chain | No | Same metaphor family, so it **must be cited**; not parametric |
| FedNL | 2605.16350 | Federated Titans | None | — | Yes | No security; it is our substrate |
| HiSPA / CLASP | 2601.01972 / 2603.12206 | SSM hidden state | Detector (empirical) | No | No | Single model, no bound |
| Krum / TrMean / FLTrust / RFA / CClip | — | FL gradients | Byzantine rates | — | Yes | Aggregate fixed-model gradients, not inner-loop memory; no source tiering; no action certificate |
| Memory-poisoning survey | 2604.16548 | — | — | — | — | Reports no certified defence for parametric memory |

**Honest novelty rating:** high for the intersection of parametric memory, quantitative certificates and federation. Not high for the individual ingredients: corroboration, trimmed means and quarantine all exist. The paper must state this openly and win on the triad *necessity (Prop. 0) → sufficiency (Thms 1–3, Prop. R) → tightness (Thm 4)* and on the evaluation.

---

## 6. Pre-registered hypotheses v2

These are frozen in `docs/preregistration.md` and committed before the full runs.

* **H1 (persistence, revised Day 2).** Lemma 1 upper-bounds the measured half-life (counterfactual twin) and is exact on orthogonal streams. On random streams the mean-field Lemma 1′ (self-correction λ = 2θ/d_k) predicts it with log-log correlation ≥ 0.95. Momentum lengthens persistence only on quiet keys and shortens it in busy memories (Day 2: r = 0.9996 over 42 linear cells).
* **H2 (defence).** At matched clean recall (≤ 1 point), Tabayyun-Cert reduces targeted ASR by ≥ 50% relative to Titans/FedAvg, and by ≥ 25% relative to the best robust-aggregation baseline, for f/n ≤ 0.2, **including the adaptive attacker**.
* **H3 (certificate).** Zero certificate violations across all runs. The runtime certificate is within 10× of the measured worst-case drift on IID and mild non-IID. The certified-action fraction is ≥ 60% on the O-RAN replays.
* **H4 (negative regime).** Under Dirichlet(0.1) heterogeneity, or with client-unique facts, high-risk benign recall drops by > 10 points and the certificate becomes vacuous.
* **H5 (necessity, new).** Every single-tier baseline (no source separation, any norm filter) reaches ASR ≥ 90% under rayn, at per-write norm ≤ 10% of the single-shot attack.
* **H6 (corroboration matters, new).** In the sparse-observation regime (ρ ≤ 0.3), trimmed mean / median / Krum over all sources lose ≥ 20 points of benign recall, or ≥ 20 points of ASR, relative to Tabayyun-Cert.

---

## 7. Experimental plan v2 (single RTX 5080, 16 GB)

**Tier A, controlled testbed (CPU/GPU minutes).** Synthetic multi-source fact streams with known honest spread, observation rate ρ, unique facts and Dirichlet heterogeneity. Used for H1, H3, H5, H6 and every ablation. ≥ 10 seeds.

**Tier B, learned memories.**
* B1: a Titans-style LMM (MAC-lite, 20–60M parameters) trained from scratch on MQAR (Zoology protocol) and a small language mix. About 4–8 GPU-hours per model in bf16.
* B2: memory retrofit. A neural-memory adapter on a frozen SmolLM2-360M or Qwen2.5-0.5B, trained on BABILong-style QA and evaluated on BABILong qa1–qa5 (4k–32k) and on CounterFact injections. About 6–10 GPU-hours.

**Tier C, O-RAN.** ColO-RAN and COMMAG KPI traces (wineslab) feed a memory-augmented slicing/scheduling policy whose sources are E2 KPI streams, xApp reports and tickets. Attacks are spoofed KPIs and poisoned tickets. Metrics: action-flip rate, certified-action fraction, KPI regret.

**Tier D (optional).** TeleQnA network copilot with retrofitted memory.

**Attacks.** single-shot, targeted, rayn, HiSPA-style amnesia, MINJA/AgentPoison-style bridging, edge-of-range (omniscient), unbounded, median-shift (liveness), gradient-based adaptive (differentiable gate relaxation, PGD), federated crafted Δ^P, basis-rotation, correlated feed, and Sybil beyond f.

**Baselines.** Titans (ungated), Titans with clip / TTL / DP-noise, Miras-Yaad (Huber), quarantine with {mean (FedAvg), TrMean, median, Krum, Multi-Krum, GeoMed/RFA, CClip, FLTrust}, CLASP-style detector, AM-Sentry, SMSR (ported), TMA-NM-style corroboration (ported), RobustRAG isolate-then-aggregate (ported), ISNAD routing (adapted), and the v1 additive rule.

**Statistics.** Paired bootstrap 95% CIs. Holm correction on the two primary contrasts (H2 vs Titans, H2 vs best robust baseline). ≥ 10 seeds on Tier A and ≥ 5 on Tiers B and C. Every number is traceable to a commit through `results/registry.jsonl`.

---

## 8. Risks (updated)

| Risk | Mitigation |
|---|---|
| Robust-aggregation baselines already win in easy regimes (Day-0 smoke: with f/n = 0.2 and ρ = 0.7, quarantine + mean already blocks the targeted write) | Pre-registered H6. Report the regime map (ρ, f/n, attack) honestly. The paper's claim is *certified* and *adaptive-robust*, not "best everywhere". |
| Certificate loose in deep memories | Deep quarantine with a linear certified tier makes it exact. Report tightness ratios. |
| "Just robust aggregation" | Necessity result, source separation inside a recurrent memory, action-level certificates; ablations isolate each component. |
| TMA-NM priority on corroboration-gating | Cite prominently. Our delta is parametric memory + quantitative certificates + federation. |
| A single GPU limits scale | Retrofit on frozen backbones. 1B-scale confirmation runs go to the partner group. |

---

## 9. Recommended framing paragraph (unchanged in spirit)

"Our design principle, *verify a consequential report from an unvetted source before acting on it, and confirm it through independent sources*, is motivated by Qur'an 49:6 and its classical exegesis. We use it purely as conceptual motivation, in the same way neuroscience motivates memory architectures. All claims are established by the theorems and experiments that follow."
