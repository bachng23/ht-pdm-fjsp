# Shared-action PPO architecture experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED
- Version Label: shared_action_ppo_v1

## Experiment overview

- **Objective**: test whether a shared per-action scorer reduces seed-dependent
  policy modes relative to the fixed-logit MaskablePPO actor.
- **Hypothesis**: shared scoring lowers mean validation objective and its
  dispersion across independent training seeds; entropy regularization further
  reduces action collapse.
- **Type**: multi-seed RL training and fresh-panel validation.

## Locked design

- Conditions: existing fixed-logit PPO, shared scorer, and shared scorer with
  `ent_coef=0.01`.
- The fixed-logit condition reuses the completed five-seed source run. Both
  shared conditions train from the same seeds 10000–14000 for 500,000 requested
  timesteps.
- Shared actor: one network scores every action from global context and that
  action's dynamic features. The critic remains separate and sees the full state.
- Validation seeds: 41000–41199, unused by all previous diagnostics.
- Reserved future test seeds: 50000–50099 remain unopened.
- Primary endpoint: validation objective averaged within each training seed.
- Secondary endpoints: across-seed dispersion, failure, maintenance, cost,
  tardiness, makespan and validation episode win rate.
- Stopping rule: exactly 500,000 requested timesteps per new model, no adaptive
  retry, checkpoint selection, early stopping or test evaluation.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| `architecture_manifest.json` | JSON | status `COMPLETED`; 10 trained cells |
| `*/train_seed_*/maskable_ppo.zip` | SB3 model | all shared models loadable |
| `architecture_episodes.csv` | CSV | 3 × 5 × 200 validation episodes |
| `architecture_summary.json` | JSON | training-seed-level comparisons |

Partial episode data and completed training cells are persisted after every
model. The runner never overwrites a non-empty run directory and uses `tqdm`.

## Ubuntu lab execution

The failure-sensitive steps run inside a subshell. A failed command therefore
returns control to the interactive terminal instead of closing its tab.

```bash
cd "$HOME/ht-pdm-fjsp"
RUN_ID="shared_action_ppo_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs
LOG="artifacts/lab_logs/$RUN_ID.log"
set -o pipefail

(
  set -euo pipefail
  test ! -e "artifacts/$RUN_ID"
  if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi
  command -v uv
  git fetch origin
  git switch codex/shared-action-ppo 2>/dev/null || \
    git switch --track origin/codex/shared-action-ppo
  git pull --ff-only
  uv sync --frozen --extra dev
  test -f artifacts/ppo_replicates_20260920T110405Z/run_manifest.json
  uv run ht-pdm-fjsp-shared-action-ppo \
    --source-run artifacts/ppo_replicates_20260920T110405Z \
    --profile full \
    --device cpu \
    --output-dir "artifacts/$RUN_ID"
) 2>&1 | tee "$LOG"

STATUS=$?
echo "EXIT_STATUS=$STATUS"
echo "LOG=$LOG"
```

Pull every lab artifact back to the Mac with one resumable command:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

This is model development on one benchmark instance. Only an architecture and
hyperparameter setting locked after validation may be evaluated once on the
reserved future test panel.
