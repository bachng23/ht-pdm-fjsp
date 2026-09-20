# Minimal HT-PdM-FJSP validation report

## Material Passport

- Artifact: experiment result and reproducibility validation
- Verification status: VERIFIED
- Environment: CPython 3.12.13, uv lockfile, NumPy 2.5.3, pytest 8.4.2
- Configuration: `configs/minimal_benchmark.json`
- Evaluation panel: seeds 0–99, paired across two policies
- Scope: implementation/mechanism validation only; not confirmatory evidence

## Automated verification

- `uv run pytest -q`: **11 passed**.
- `uv run python -m compileall -q src tests`: passed.
- Two independent 100-seed benchmark runs produced byte-identical directories
  under recursive `diff`.
- Every episode passed the built-in feasibility auditor for operation
  completeness and precedence, machine eligibility, machine non-overlap,
  technician non-overlap, technician eligibility, and maintenance age reduction.

## 100-seed smoke results

| Metric (mean) | Production-first SPT | Health-threshold SPT |
|---|---:|---:|
| Failures | 2.08 | 0.61 |
| Makespan | 36.13 | 37.25 |
| Total tardiness | 11.23 | 12.07 |
| Maintenance waiting | 0.28 | 1.23 |
| Total cost | 76.63 | 72.20 |

Paired mean differences (`health-threshold − production-first`) were −1.47
failures, +1.11 makespan, +0.84 tardiness, +0.96 maintenance waiting, and −4.43
total cost. Health-threshold produced fewer failures on 87/100 seeds and lower
weighted cost on 52/100 seeds.

## Interpretation boundary

The smoke panel confirms that all intended couplings activate and expose a
meaningful trade-off: preventive maintenance reduces failures and mean weighted
cost but occupies scarce skilled technicians and slightly increases production
time. These numbers do not establish algorithmic superiority because the
parameters are synthetic, the cost weights are provisional, and the threshold
policy observes latent effective age.

## Gymnasium and advanced baselines extension

- `uv run pytest -q`: **20 passed**, including the Gymnasium checker, dynamic
  mask/resource exclusivity, safe invalid actions, CP-SAT feasibility, reward
  identity, complete trace audits, exact rollout reproducibility, and a test
  that scenario lookahead is independent of the episode's future seed.
- Five policies × 100 common seeds completed without truncation; outputs are in
  `artifacts/gym_baselines`.
- Two independent 10-seed artifact runs were byte-identical under recursive
  `diff` (config, metadata, episode table, paired comparisons, summary, traces).

| Policy | Objective mean ↓ | Makespan mean ↓ | Failures mean ↓ | Win rate vs. masked SPT |
|---|---:|---:|---:|---:|
| Masked SPT | 113.07 | 36.10 | 2.09 | — |
| CP-SAT reactive | 106.05 | 35.93 | 1.20 | 59% |
| Health threshold | 99.18 | 34.82 | 0.95 | 72% |
| Rolling horizon | 93.10 | 33.59 | 1.06 | 73% |
| Joint risk greedy | **89.25** | **32.78** | 1.07 | **74%** |

The rolling-horizon policy evaluates independent common planning scenarios and
uses a conservative improvement margin over its joint-risk base policy. It does
not clone the evaluated seed's unrevealed failure sequence. CP-SAT optimizes the
deterministic production plan and handles maintenance reactively; it is not an
integrated stochastic optimal solver.

These are benchmark smoke results on one small synthetic instance, not evidence
of general superiority. Hyperparameters and cost weights require train/tuning,
validation, and held-out test instance sets before use in a paper comparison.

## MaskablePPO local smoke

- Profile: 512 training steps, seed 10000, five validation seeds, five held-out
  test seeds; macOS CPU.
- Status: **COMPLETED**; model checkpoint, Monitor CSV, SB3 progress CSV,
  episode CSV, summary, and runtime manifest were produced.
- PPO test objective was 85.0 on this five-seed smoke panel versus 106.86 for
  masked SPT. This tiny panel validates integration only and is not a performance
  conclusion.
- PPO evaluation produced no invalid actions, every trace passed the feasibility
  audit, and every episode satisfied the reward/objective identity.
