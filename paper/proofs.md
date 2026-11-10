# Proofs (working draft; moves to `proofs.tex` for the paper)

Notation: linear memory M ∈ R^{d_v×d_k}, unit keys, attentional bias ℓ = ‖Mk − v‖², gradient g = 2(Mk − v)kᵀ, Titans update S_t = ηS_{t−1} − θ_t g_t, M_t = (1−α)M_{t−1} + S_t. The per-write filter clips ‖g‖ ≤ G.

---

## Lemma 1 (single-write persistence, exact on orthogonal streams)
Let the write at t = 0 have gradient g₀, and let every later key be orthogonal to k*. Then M_T k* − M_T^{clean} k* = −θ g₀k* · Σ_{j=0}^{T}(1−α)^{T−j}η^j.

*Proof.* By linearity, the difference D_t = M_t − M_t^{clean} and the momentum difference Σ_t = S_t − S_t^{clean} obey Σ_t = ηΣ_{t−1} − θ(g_t − g_t^{clean}), D_t = (1−α)D_{t−1} + Σ_t. For t ≥ 1 the write key k_t ⊥ k*. So g_t − g_t^{clean} = 2D_{t−1}k_t k_tᵀ, and (g_t − g_t^{clean})k* = 2D_{t−1}k_t(k_tᵀk*) = 0. Hence Σ_t k* = ηΣ_{t−1}k* = η^tΣ_0k*, and D_T k* = Σ_{t=0}^T(1−α)^{T−t}Σ_t k*. ∎ (Numerically exact, Day 3: 42/42 cells, zero error.)

## Lemma 1′ (mean-field persistence with benign self-correction)
Let later keys be i.i.d. uniform on the unit sphere, independent of the past. Write x_t = E[D_t k*] and s_t = E[Σ_t k*]. Then
  s_t = η s_{t−1} − λ x_{t−1},  x_t = (1−α)x_{t−1} + s_t,  λ = 2θ/d_k.

*Proof.* (g_t − g_t^{clean})k* = 2D_{t−1}k_t k_tᵀk*. Since k_t is independent of D_{t−1}, E[k_t k_tᵀ] = I/d_k, so E[2D_{t−1}k_tk_tᵀk*] = (2/d_k)E[D_{t−1}k*]. Substitute this into the recursions of Lemma 1. ∎

*Remarks.* (i) λ acts like an extra forget gate: busy memories self-correct poison. (ii) The momentum recursion is driven by −λx, so **momentum amplifies self-correction**. The eigenvalues of the 2×2 system shrink as η grows for small α, which is why persistence *decreases* with η in busy memories (Day 2: 272 → 32 steps for η = 0 → 0.95). (iii) The persistence in the actual (non-expected) stream concentrates around the mean field: Day 2 found 42/42 cells within −9%…+4% (log-log r = 0.9996). (iv) **Deep memories.** For the 2-layer MLP M(k) = W₂σ(W₁k), the forget gate scales both layers, so the read-out decays as (1−α)² ≈ 1 − 2α. With α_eff = 2α and one interference constant fitted on the η = 0 cells only, the law predicts the 35 held-out MLP cells with log-log r = 0.994 (Day 3, exploratory). Conjecture: for depth L, α_eff = Lα.

## Proposition 0 (necessity of source separation)
Consider a single-tier linear memory with any per-write filter G > 0 (also G = ∞), constant gates, and an established association M k* ≈ v_true that one honest source refreshes once every R steps. One adversarial source that writes to k* at least once every r < R steps, with writes whose gradient norm ≤ G (so the filter never triggers), can make the read at k* closer to any v* than to v_true.

*Proof sketch (η = 0, G < ∞; momentum only speeds it up).* Restrict to the 2-D plane spanned by v_true and v*, and let x_t = M_t k*. An honest refresh has gradient 2(x − v_true), clipped to norm ≤ G, so it moves x by at most θG. The adversary writes with target x + (G/2)(v* − x)/‖v* − x‖. Its gradient norm is exactly G (never clipped), and it moves x by θG straight towards v*. Decay and benign writes with random keys shrink x radially (Lemma 1′) and favour neither target. In any window containing R adversarial and r < R honest writes, the projection of x on the unit direction (v* − v_true)/‖v* − v_true‖ therefore increases by at least θG(R − r)·c, where c > 0 is the cosine between the adversary's step and that direction, which stays bounded away from 0 while x is closer to v_true. Since that projection must exceed ‖v* − v_true‖/2 for the flip, the flip happens after O(‖v* − v_true‖ / (cθG(1 − r/R))) adversarial writes. ∎ For G = ∞ the claim is immediate: one unclipped write between two refreshes sets x arbitrarily close to v*, so the attacker wins at every time between its write and the next refresh.

*Interpretation.* A per-write norm filter bounds the **size** of each write equally for honest and adversarial sources, so it converts the contest into one of **write frequency**, which the attacker controls. Only a defence that counts **independent sources** rather than writes can bound a minority. Tabayyun's per-source quarantine and k-of-n promotion do exactly this (Thm 1).

*Empirical check (Day 3, 10 seeds × 2 momenta, filters G ∈ {∞, 0.5, 0.2, 0.1, 0.05, 0.02}).* Single-shot ASR is 0 under every filter, and its unfiltered flip is transient because the honest refresh restores the fact. Rayn ASR is 1.0 under every filter (20/20 runs per filter), with a median of 21–35 writes to flip, and that number barely depends on G, exactly as the proof predicts (the filter also clips the honest refresh).

## Lemma 2 (hull) and Proposition R (runtime certificate)
Fix a direction u and a coordinate i. Let x_1 ≤ … ≤ x_c be the corroborators' values, of which at most f are adversarial (f ≤ β, c ≥ 2β+1), and let H be the honest corroborators (|H| = h ≥ c − f).

*Lemma 2.* TrMean_β(x) ∈ [min_H x, max_H x]. *Proof.* Every value below min_H x is adversarial, so there are at most f ≤ β of them, and all are trimmed. The same holds above. The retained values lie in [x_{β+1}, x_{c−β}] ⊆ [min_H, max_H]. ∎

*Proposition R.* Let L = mean of the c−f smallest values and U = mean of the c−f largest. Then μ_H := mean_H x ∈ [L, U], hence |TrMean − μ_H| ≤ e := max(TrMean − L, U − TrMean). *Proof.* H is a subset of size h ≥ c−f. The mean of any h-subset is at least the mean of the h smallest values, which is at least the mean of the c−f smallest (adding larger elements cannot lower a mean). Symmetrically for U. ∎

## Theorem 4 (instance-optimality of the runtime certificate), new on Day 5
Among all bounds B(x) computed from the observed corroborator values x that are **sound** (|TrMean(x) − μ_H| ≤ B(x) for every placement of ≤ f adversaries consistent with x), Prop. R is the smallest: B(x) ≥ e(x) for every x.

*Proof.* Given x, both "the c−f smallest values are the honest ones" and "the c−f largest values are the honest ones" are consistent with the threat model (the other f values are adversarial). In these two worlds μ_H = L and μ_H = U respectively, and the observation x, and hence TrMean(x) and B(x), is identical. Soundness in both worlds requires B(x) ≥ max(TrMean − L, U − TrMean) = e(x). ∎

*Remarks.* (i) The bound is exactly attained, so it cannot be improved by any constant. (ii) The theorem is per coordinate. Jointly over coordinates and directions the adversary must use one identity set, so the joint bound can be loose in principle. Empirically the omniscient edge-of-range attacker attains it: Day 5 shows true drift / certificate = 0.94–0.98 (median) and up to 0.998 (Frobenius), and up to 1.0000003 (float32) at the attacked key. Non-adaptive attacks reach a median of 0.29, so the certificate costs a factor of about 3.5 against weak attackers and nothing against the strongest one.

## Theorem 1, Theorem 2′, Theorem 3: see `docs/proposal_v2.md` §4; full write-up on Day 7.
