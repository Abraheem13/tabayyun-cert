# Pre-registration (DRAFT, to be frozen on Day 7 by a tagged commit `prereg-v1`)

Once tagged, this file can only change by appending an **amendment** with a date and a reason. Results that contradict a hypothesis are reported, never dropped.

## Primary contrasts (Holm-corrected, α = 0.05)
1. **P1**: targeted ASR, Tabayyun-Cert vs Titans/FedAvg, f/n ∈ {0.1, 0.2}, pooled over attacks incl. adaptive. Success: relative reduction ≥ 50%.
2. **P2**: targeted ASR, Tabayyun-Cert vs best robust-aggregation baseline (chosen *per attack* as the strongest baseline, so the comparison is against the hardest one). Success: relative reduction ≥ 25%.

Utility constraint for both: clean high-risk recall within 1 point of the comparator (paired bootstrap 95% CI).

## Secondary hypotheses
H1, H3, H4, H5, H6 exactly as in `docs/proposal_v2.md` §6.

## Pre-declared negative regimes (expected failures, will be reported)
* Dirichlet(0.1) source/site heterogeneity.
* Client-unique facts (never promoted, by design).
* Correlated upstream feed and Sybils beyond f (out of threat model).

## Fixed analysis choices
* Seeds: Tier A 10, Tier B 5, Tier C 5. Seeds 0..N−1, never cherry-picked.
* CIs: paired bootstrap, 10,000 resamples.
* ASR definition: flip if cos(read, v*) > cos(read, v_true); insertion if cos(read, v*) > 0.5.
* Recall: nearest-candidate accuracy over all fact values plus the attacker target.
* Certificate violation: any run with true drift > certificate·(1+1e−4)+1e−5 counts as a violation (float32 tolerance).
* Hyper-parameters (k, γ, ν, τ, α^Q, α^P) are tuned **only on the clean validation split, never on attacked runs**.
