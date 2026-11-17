# IEEE TIFS: submission rules and house style (read 2026-11-17)

Sources: IEEE Signal Processing Society, "Information for Authors" (SPS Transactions & Journals) and the TIFS journal page; T-IFS "Guides for deep learning submissions" (PDF). IEEE Xplore (punumber 10206) blocks automated readers, so the issue page itself was not read. The house style below is taken from Asres, Jiao and Omlin, *Low-Latency Video Anonymization for Crowd Anomaly Detection*, IEEE TIFS vol. 21, 2026 (DOI 10.1109/TIFS.2025.3630347).

## Rules (quoted numbers)
- **Scope:** information forensics, information security, biometrics, surveillance, and systems applications that incorporate these features.
- **Type:** Regular Paper ("complete disclosure of a technical premise").
- **Length at first submission: ≤ 13 double-column pages, 10-pt font,** *including* title, authors, abstract, text, figures, tables, **appendices and proofs, and references**. Supplemental material and graphical abstract are not counted.
- **Revised manuscript: ≤ 16 pages.** Appendices and proofs then move to supplemental material and are not counted.
- **Supplemental material:** recommended ≤ 6 double-column pages (10-pt). Numbering of equations, theorems, figures and tables must continue that of the paper. Proofs there may not contain new theorems that the paper does not reference.
- **Format:** IEEE template (LaTeX `IEEEtran`, journal mode), double column, single spaced, 10-pt font, margins ≥ 1 inch.
- **Abstract: 150–250 words**, self-contained, **no abbreviations, footnotes, displayed equations or references**, and it should include a few keywords.
- **Index Terms** required. **EDICS** category required at submission (SPS Unified EDICS).
- **ORCID required for all authors.** Submission through the IEEE Author Portal. The author list must match the portal's "Authors and Institutions".
- **Review:** single-anonymized; at least 2 independent reviewers; screened for plagiarism.
- **Immediate rejection if:**
  - novelty is lacking ("a straightforward combination of well-established … theories and algorithms");
  - experimental data are insufficient;
  - the English is poor or the typos are too many;
  - citations are insufficient.
- **The manuscript must state:**
  - the problem;
  - the contribution;
  - why it is significant;
  - the closest literature;
  - what is new relative to that literature.
- **Preprints:** arXiv is allowed if identical to the submission. List all posted preprints (with URLs) at submission and do **not** cite the identical preprint. Non-identical preprints count as prior work and must be cited and explained.
- **Resubmission** after rejection: once only, with verbatim reviewer quotes and responses.
- **No changes after acceptance.** The "Formatted (Double Column) Main File" PDF must be publication-ready at every stage; remove coloured text and annotations in revisions.
- **Charges:**
  - $110/page for the first 10 pages (voluntary);
  - **$220/page for every page beyond 10 (mandatory)**, which also applies to open-access articles;
  - hybrid open access with the IEEE APC;
  - colour is free (online-only).
- **Reproducibility:** strongly encouraged (code and data that regenerate every figure and table). The T-IFS deep-learning checklist asks for:
  - network type and topology, loss, and pre-processing;
  - initialisation, optimiser and hyperparameters, batches, learning-rate schedule, stopping criterion, and **which data set the hyperparameters**;
  - train / validation / test split and cross-validation;
  - GPU model, tool versions;
  - **statistical significance of claimed results**.

## House style of the reference TIFS paper (to mirror)
- Title of the form "Main idea: trade-off", with author affiliations and IEEE membership grades. The first-page footnote carries the received / revised / accepted dates, funding and the corresponding author.
- Abstract ending with "Code: <GitHub URL>", then Index Terms.
- **Acronym table and notation table on page 1** (two-column lists of symbols with meanings).
- Introduction: motivation, gap, bullet list of contributions, and a roadmap paragraph.
- Sections:
  - II Literature Review;
  - III Methodology (A Problem Formulation with numbered equations; B Method; C/D component models);
  - IV Results and Discussion (A Evaluation approach and metrics; B, C main evaluations; D Computational cost);
  - V Conclusion;
  - Appendix (supplementary results, benchmark comparison table, limitations and future directions);
  - References;
  - author biographies with photos.
- Formal problem formulation with `arg max` / `arg min`, sets and operators defined in the text right after each equation.
- **Numbered algorithms** (Algorithm 1–4) in `algorithmic` style with comments on every line.
- **System pipeline block diagrams** (two variants: baseline vs proposed), plus qualitative image grids.
- Tables:
  - small caption in capitals, ↑/↓ arrows for the better direction, best in **bold**;
  - relative change columns (Δ against baseline and against the non-adaptive variant);
  - footnotes marking numbers reported by other papers (*).
- Scatter plots of the trade-off (privacy vs utility) across all methods, and per-class bar charts.
- A computational-cost analysis (time and memory, CPU and GPU, hardware stated).
- A benchmark positioning table (method × capability with ✓/✗).
- Limitations stated explicitly, as bullets with mitigations.
- References: about 58, IEEE style, recent (2023–2025) and including prior TIFS papers.

## Implications for our manuscript
- 13 pages *including* proofs at submission. The core theorems (Thm 6, Prop. R/R′, Thm 3a′/3b, Thm 5) go in the main text with short proofs. Long proofs, the ablation tables and tier-by-tier details go to ≤ 6 pages of supplemental material.
- EDICS: choose from the security / forensics categories (e.g. "Secure and trustworthy machine learning" / adversarial machine learning) when submitting.
- Cite ≥ 2–3 recent TIFS papers on poisoning / Byzantine-robust aggregation / memory or RAG security.
- Report statistical significance (bootstrap CIs, Holm) for every claimed difference, and the hyperparameter-selection data (our clean-only tuning rules), as the T-IFS checklist requires.
