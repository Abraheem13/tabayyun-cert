# Tabayyun-Cert

**Certified, corroboration-gated quarantine for poisoned test-time memory in federated, agentic AI-native networks.**

Every write from an unvetted source lands in a per-source quarantine memory. Only content that at least *k* independent sources corroborate is promoted, via a trimmed mean, into the shared, action-driving memory. A run-time-computable certificate bounds how far ≤ *f* malicious sources can move that memory, and whether they can flip a network action.

> Design principle (conceptual motivation only): *verify a consequential report from an unvetted source before acting on it, and confirm it through independent sources* (Qur'an 49:6 and its classical exegesis). All claims rest on the theorems and the pre-registered experiments.

- Proposal v1: [`docs/proposal_v1.pdf`](docs/proposal_v1.pdf). Proposal v2 (12 fixes): [`docs/proposal_v2.md`](docs/proposal_v2.md)
- Plan: [`ROADMAP.md`](ROADMAP.md). Daily log: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Ablations: [`ABLATIONS.md`](ABLATIONS.md). Changes: [`CHANGELOG.md`](CHANGELOG.md)
- Theory status: [`docs/theory_status.md`](docs/theory_status.md). Pre-registration: [`docs/preregistration.md`](docs/preregistration.md)

## Layout
```
src/tabayyun/
  memory.py      Titans-style linear / MLP test-time memories (batched, per-source)
  gate.py        tabayyun gate (corroboration + trimmed mean + runtime certificate), FL aggregators
  tiered.py      TabayyunMemory: quarantine tier + linear certified promoted tier (EMA / additive)
  certify.py     Lemma 1, Thm 2', Prop. R, Thm 3 (action), Thm 3b (delay law)
  attacks.py     targeted, single-shot, rayn, edge-of-range, unbounded, median-shift
  baselines.py   Titans (+clip/TTL/DP), quarantine + {mean, TrMean, median, Krum, GeoMed, CClip}
  data/streams.py controlled multi-source fact world (rho, sigma, unique facts, Dirichlet)
  metrics.py, registry.py (every run -> results/registry.jsonl with git commit)
experiments/     e01_persistence.py (H1, H5), e02_synthetic_defence.py (H2, H3, H6), ...
tests/           numerical checks of every theorem (a violated certificate = a bug)
```

## Quick start (RTX 5080)
Windows PowerShell (from the repo root):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\python.exe scripts\run_day.py 1
```
Linux / WSL2:
```bash
bash scripts/setup_env.sh && source .venv/bin/activate && python scripts/run_day.py 1
```
`run_day.py` runs the tests and that day's experiments, and saves all output to `results/dayNN/console.txt`.
The RTX 5080 (Blackwell, sm_120) needs PyTorch >= 2.7 built with CUDA 12.8+ (`--index-url https://download.pytorch.org/whl/cu128`); `scripts/check_gpu.py` verifies this and writes `results/env.txt`.

## Rules of the repo
1. Every number in the paper comes from `results/registry.jsonl` (commit and config hash).
2. Smoke runs are never results.
3. Hyper-parameters are tuned on clean validation only. Hypotheses are frozen in `docs/preregistration.md`.
4. Negative results are logged and reported.
