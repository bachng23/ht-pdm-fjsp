# Validation-selected PPO checkpoint experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED
- Version Label: ppo_checkpoint_selection_v1

## Experiment overview

- **Objective**: test whether validation-only checkpoint selection improves the
  stability and held-out objective of the existing five PPO training replicates.
- **Hypothesis**: the mean held-out objective of validation-selected checkpoints
  is lower than that of the predeclared final 500k models.
- **Type**: model selection and held-out simulation evaluation; no retraining.

## Locked design

- Source run: a completed `ht-pdm-fjsp-ppo-replicates` artifact directory.
- Selection data: validation seeds 20000–20019 only.
- Selection rule: minimum mean validation objective independently for each
  training seed; exact ties select the earlier checkpoint.
- Test data: held-out seeds 30000–30099, opened only after selection is fixed.
- Primary endpoint: selected-minus-final objective averaged within each training
  seed, with independent training seed as the replication unit.
- Secondary endpoints: makespan, tardiness, failures, maintenance, total cost,
  baseline comparisons, and episode-level win rate.
- Stopping rule: evaluate every selected checkpoint once on the fixed test panel;
  there is no adaptive stopping or retry.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| `selection_manifest.json` | JSON | status `COMPLETED`; disjoint validation/test panels |
| `selected_checkpoints.csv` | CSV | one validation-selected checkpoint per training seed |
| `selected_ppo_episodes.csv` | CSV | 5 × 100 held-out episodes |
| `selection_summary.json` | JSON | paired selected-vs-final and baseline summaries |

The runner writes a partial episode CSV after every model, never silently
overwrites a non-empty output directory, and shows a `tqdm` progress bar.

## Ubuntu lab execution

The full evaluation reuses the completed replication run and does not train new
models:

```bash
git fetch origin
git switch --track origin/codex/ppo-checkpoint-selection
git pull --ff-only
uv sync --frozen --extra dev

test -f artifacts/ppo_replicates_20260920T110405Z/run_manifest.json
test -f artifacts/ppo_replicates_20260920T110405Z/checkpoint_validation.csv

set -o pipefail
RUN_ID="ppo_checkpoint_select_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs

uv run ht-pdm-fjsp-ppo-select \
  --source-run artifacts/ppo_replicates_20260920T110405Z \
  --device cpu \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/lab_logs/$RUN_ID.log"
```

Pull all artifacts to the Mac with resumable progress:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

Checkpoint selection is a legitimate optimization because no test outcome is
used by the selection rule. The experiment still covers one synthetic problem
instance and does not establish cross-instance generalization.
