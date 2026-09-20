# PPO failure-mode diagnostic plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED
- Version Label: ppo_diagnostics_v1

## Experiment overview

- **Objective**: identify whether PPO training-seed instability is associated
  with value fitting, entropy collapse, update instability, action masking, or
  maintenance/risk behavior.
- **Hypothesis**: the two weak training seeds exhibit a reproducible diagnostic
  signature that differs from the three stronger seeds.
- **Type**: post-training learning-dynamics analysis and fresh-seed simulation.

## Locked design

- Reuse the five final models and their untouched training logs; do not retrain.
- Diagnostic seeds: 40000–40199, disjoint from every prior split.
- Reserved future test seeds: 50000–50099; the diagnostic runner must not open
  this panel.
- Primary diagnostics: late explained variance, entropy, KL, clip fraction,
  reward trajectory, action mix, mask size, selected production failure risk,
  maintenance opportunities/take rate, reward decomposition, failures and cost.
- Replication unit remains the independent PPO training seed. Diagnostic
  episodes characterize a fixed policy and are not independent training runs.
- Stopping rule: exactly 200 diagnostic episodes per model, without adaptive
  retry, checkpoint selection, or early stopping.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| `diagnostic_manifest.json` | JSON | status `COMPLETED`; future test unopened |
| `training_dynamics.csv` | CSV | one complete row per training seed |
| `diagnostic_episodes.csv` | CSV | 5 × 200 complete episodes |
| `diagnostic_decisions.csv` | CSV | every decision from the 1,000 episodes |
| `diagnostic_summary.json` | JSON | per-training-seed behavior and reward summaries |

Partial episode and decision tables are written after every model. The runner
does not overwrite a non-empty output directory and displays one `tqdm` bar.

## Ubuntu lab execution

```bash
set -euo pipefail

git fetch origin
git switch --track origin/codex/ppo-diagnostics
git pull --ff-only
uv sync --frozen --extra dev

test -f artifacts/ppo_replicates_20260920T110405Z/run_manifest.json
test -f artifacts/ppo_replicates_20260920T110405Z/aggregate_summary.json

RUN_ID="ppo_diagnostics_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs

uv run ht-pdm-fjsp-ppo-diagnostics \
  --source-run artifacts/ppo_replicates_20260920T110405Z \
  --device cpu \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/lab_logs/$RUN_ID.log"
```

Pull all results back to the Mac:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

Correlations across five training seeds are exploratory. A diagnostic signature
must motivate a predeclared intervention and a new multi-seed experiment; it is
not itself evidence that the intervention works.
