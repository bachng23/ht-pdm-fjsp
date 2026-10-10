# Maintenance opportunity cost v1 — macOS engineering gate

Date: 2026-10-10 (Asia/Taipei). Branch:
`codex/maintenance-opportunity-cost-v1`; base verified solver commit:
`4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8`.

## Verification observed

- `uv sync --frozen --extra dev` created an isolated Python 3.12.13 environment.
- Repository suite: **583 passed**, 3 nonfatal inherited warnings, 227.69 s.
- After final diagnostics/interruption changes: **12 experiment tests passed**.
- Solver-comparison plus new experiment integration suite: **17 passed** before
  the two additional interruption/capacity tests were added.
- One N2 engineering state at exact horizon 8 fits the 250,000-state cap.
- No full experiment, remote execution, SSH or transfer was performed.

Final CLI smoke directory:
`artifacts/maintenance_opportunity_cost_v1_smoke_20261010T073700Z/`.
Its console log is the same stem with `.log` beside the directory.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --frozen python -m ht_pdm_fjsp.maintenance_opportunity_cost \
  --profile smoke --device cpu \
  --output-dir artifacts/<new-UTC-timestamped-smoke-directory>
```

The recorded invocation used `--no-sync` after the frozen sync and a temporary
uv cache. All three tqdm stages appeared: exact values, development evaluation,
paired multi-seed evaluation. Training/checkpoint reload is NOT APPLICABLE.

## Artifact audit

Manifest: COMPLETED; audit PASS; `full_test_outcomes_opened=false`.

| Output | Verified rows |
|---|---:|
| development_episodes.csv | 84 |
| episodes.csv / episodes.partial.csv | 30 / 30 |
| decisions.csv / latency.csv | 120 / 120 |
| coordination.csv | 30 |
| action_values.csv | 2 |

All manifest artifact SHA-256 hashes matched. Pooled decision counts and
request counts reconcile with raw CSVs. Integration tests replay logged
scalar transitions and costs; runner independently reconciles each physical
tick with the vectorized transition. Request waiting/censoring and failed
manifest persistence passed. Nonempty output directories are rejected.

The smoke snapshot was created on the base commit with uncommitted new
experiment source; snapshot hash was checked against the source being
committed. This report is engineering evidence, not a scientific outcome.
Smoke's mechanism/result screens are ENGINEERING_ONLY; no conclusion about
full H6/H8 mechanisms or N4/H12 assignment headroom follows from it.

## Full scope to hand off

32 development cohorts x 8 shocks x 3 conditions x 7 rule candidates:
5,376 development episodes. 64 new test cohorts x 32 shocks x 3 conditions
x 5 roles: 30,720 test episodes, 368,640 logged physical decisions.
Exact grid: 48 conditional assignment states; 128 forecast scenarios per
planner decision. Each attempt needs a new UTC artifact directory.

Expected output is several hundred MB, potentially around 1 GB depending
on CSV serialization and request counts; no measured full-run time estimate
is claimed from the short smoke. CPU only, one BLAS/OpenMP thread; keep the
lab terminal open. Follow the exact final lab and rsync handoff commands.
