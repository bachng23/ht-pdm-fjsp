# MaskablePPO multi-seed smoke report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-09-20
- Verification Status: VERIFIED
- Version Label: ppo_multiseed_smoke_v1

## Execution

- Platform: local macOS CPU.
- Training: two independent seeds, 256 steps each.
- Evaluation: two validation seeds and three held-out test seeds per model.
- Checkpoints: four per model, all evaluated on validation seeds.
- Baselines: masked SPT, health threshold, joint risk, and reactive CP-SAT.

## Validation result

- Full test suite: 26 passed before the smoke run.
- Both models and all expected CSV/JSON/model artifacts were created.
- Manifest status: `COMPLETED`; completed training seeds: 10000 and 11000.
- A second independent smoke execution produced byte-identical PPO episode,
  checkpoint-validation, baseline, and aggregate-summary artifacts.
- The two under-trained policies differed substantially, confirming that the
  full experiment must treat training seed—not episode seed—as the primary
  replication unit.

This smoke validates orchestration and reproducibility only. Its 256-step policy
metrics are not an algorithm-performance result.
