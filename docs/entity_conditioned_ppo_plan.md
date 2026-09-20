# Entity-conditioned shared-action PPO experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED
- Version Label: entity_conditioned_ppo_v1

## Experiment overview

- **Objective**: test whether linking each action to the state of its referenced
  job, machine, and technician improves the shared scorer without restoring the
  seed instability of fixed action logits.
- **Hypothesis**: entity-conditioned scoring lowers mean validation objective
  relative to the corresponding shared scorer and retains most of its reduction
  in dispersion across independent training seeds.
- **Type**: controlled architecture ablation with independent PPO replicates.

## Locked design

- Source conditions: fixed logits, shared scorer, and shared scorer with
  `ent_coef=0.01` from the two completed source runs.
- New conditions: entity-conditioned shared scorer, with and without
  `ent_coef=0.01`.
- Entity context contains the linked job state, machine state, technician state,
  presence indicators, and normalized operation position. The action scorer is
  still shared across action rows.
- Training seeds: 10000–14000 for both new conditions.
- Training budget: 500,000 requested steps per new model, with the same PPO
  settings as the source experiment.
- Validation seeds: 42000–42199, unused by earlier training, validation,
  diagnostics, or route-guard experiments.
- Reserved test seeds: 50000–50099 remain unopened.
- Primary endpoint: mean validation objective within each training seed for
  entity scorer minus its matching non-entity shared scorer.
- Secondary endpoints: across-seed dispersion, makespan, tardiness, failure,
  maintenance, cost, and validation-episode win rate.
- Stopping rule: fixed training budget; no early stopping, retry, checkpoint
  selection, or test evaluation.

## Expected outputs

| Output | Success criterion |
|---|---|
| `entity_manifest.json` | `COMPLETED`; 10 new training cells; test unopened |
| `benchmark_config.json` | exact config snapshot with SHA-256 in manifest |
| `entity_episodes.csv` | 5 × 5 × 200 = 5,000 unique rows |
| `entity_summary.json` | training-seed-level and paired ablation summaries |
| `entity_*/train_seed_*/maskable_ppo.zip` | all 10 models loadable |

Partial episode data and completed training cells are written after each model.
The runner refuses a non-empty output directory unless overwrite is explicit and
uses horizontal `tqdm` progress bars without SB3 metric tables.

## Ubuntu lab execution

```bash
cd "$HOME/ht-pdm-fjsp"
RUN_ID="entity_conditioned_ppo_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs
LOG="artifacts/lab_logs/$RUN_ID.log"
set -o pipefail

(
  set -euo pipefail
  test ! -e "artifacts/$RUN_ID"
  if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi
  command -v uv
  git fetch origin
  git switch codex/entity-conditioned-ppo 2>/dev/null || \
    git switch --track origin/codex/entity-conditioned-ppo
  git pull --ff-only
  uv sync --frozen --extra dev
  test -f artifacts/ppo_replicates_20260920T110405Z/run_manifest.json
  test -f artifacts/shared_action_ppo_20260920T123821Z/architecture_manifest.json
  uv run ht-pdm-fjsp-entity-conditioned-ppo \
    --fixed-source-run artifacts/ppo_replicates_20260920T110405Z \
    --shared-source-run artifacts/shared_action_ppo_20260920T123821Z \
    --profile full \
    --device cpu \
    --output-dir "artifacts/$RUN_ID"
) 2>&1 | tee "$LOG"

STATUS=$?
echo "EXIT_STATUS=$STATUS"
echo "LOG=$LOG"
echo "RESULT_DIR=artifacts/$RUN_ID"
```

Pull all artifacts back to the Mac:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

This remains development on one synthetic benchmark instance. The reserved test
panel is opened only after the architecture and hyperparameters are frozen.
