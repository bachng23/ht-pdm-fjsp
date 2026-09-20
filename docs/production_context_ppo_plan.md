# Production-only context PPO ablation

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED
- Version Label: production_context_ppo_v1

## Locked experiment

- **Question**: can relational context improve production dispatching without the
  preventive-maintenance shortcut observed with full entity context?
- **Treatment**: shared action scorer with `ent_coef=0.01`; only production action
  rows receive linked job state, normalized operation position, and machine ID.
  Advance, preventive, and corrective rows receive an all-zero context vector.
- **Primary control**: completed `shared_scorer_entropy` models.
- **Negative control**: completed full `entity_scorer_entropy` models.
- **Training**: five independent seeds (10000--14000), 500,000 requested steps
  per new model, and the same PPO settings as the source experiments.
- **Validation**: seeds 44000--44199. Test seeds 50000--50099 remain unopened.
- **Primary endpoint**: per-training-seed mean objective difference,
  production-only context minus shared scorer entropy.
- **Stopping**: fixed budget; no early stopping, model selection, retry, or test
  evaluation.

## Expected artifacts

| File | Success criterion |
|---|---|
| `production_context_manifest.json` | `COMPLETED`; five new cells; test unopened |
| `benchmark_config.json` | exact config snapshot and matching SHA-256 |
| `production_context_episodes.csv` | 3 × 5 × 200 = 3,000 unique rows |
| `production_context_summary.json` | paired primary and negative-control results |
| `production_context_entropy/train_seed_*/maskable_ppo.zip` | five loadable models |

The runner writes partial results after every source validation or trained model,
refuses to overwrite a non-empty directory by default, and uses horizontal
`tqdm` progress bars while suppressing Stable-Baselines3 metric tables.

## Lab workflow

The final handoff supplies one failure-safe Ubuntu command to fetch this branch
and run the full experiment, followed by one `rsync` command that retrieves the
complete artifact tree to the Mac.

## Interpretation boundary

This is an architecture ablation on one synthetic benchmark instance. The
reserved test panel is not opened until the architecture is frozen.
