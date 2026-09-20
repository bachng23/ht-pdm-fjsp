# MaskablePPO multi-seed replication plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED
- Version Label: ppo_multiseed_replication_v1

## Experiment overview

- **Objective**: quantify whether the PPO advantage survives independent
  training randomness rather than only stochastic evaluation seeds.
- **Hypothesis**: mean PPO objective remains below joint-risk greedy on the
  held-out seed panel across independent training replicates.
- **Type**: training and simulation evaluation.

## Design

- Independent training seeds: 10000, 11000, 12000, 13000, 14000.
- Each replicate: 500,000 requested timesteps, four vector environments.
- Validation seeds: 20000–20019.
- Test seeds: 30000–30099, identical across every trained model and baseline.
- Baselines are evaluated once because they do not depend on training seed.
- Checkpoints at approximately 100k intervals are evaluated on validation seeds
  only; the final model remains the predeclared test model.
- Primary experimental unit: independent PPO training seed.
- Secondary sampling unit: stochastic environment test seed.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| `run_manifest.json` | JSON | status is `COMPLETED`, five seeds listed |
| `train_seed_*/maskable_ppo.zip` | SB3 model | five loadable final models |
| `ppo_episodes.csv` | CSV | 5 × (20 validation + 100 test) rows |
| `checkpoint_validation.csv` | CSV | all saved checkpoints evaluated |
| `baseline_episodes.csv` | CSV | each baseline × 100 test seeds |
| `aggregate_summary.json` | JSON | training-seed and paired summaries |

Partial CSVs and `completed_train_seeds` are updated after every replicate so a
failed long run retains completed work. The runner does not silently retry or
overwrite an existing output directory.

## Interpretation boundary

This experiment estimates training-seed stability on one synthetic problem
instance. It does not establish cross-instance or cross-scale generalization.

## Lab execution

```bash
git fetch origin
git switch --track origin/codex/ppo-multiseed-replication
git pull --ff-only
uv sync --frozen --extra dev

test -f configs/minimal_benchmark.json
test -f src/ht_pdm_fjsp/multi_seed_experiment.py

set -o pipefail
RUN_ID="ppo_replicates_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs

uv run ht-pdm-fjsp-ppo-replicates \
  --profile full \
  --device cpu \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/lab_logs/$RUN_ID.log"
```
