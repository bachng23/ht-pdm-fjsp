# MARL Factorial and Capacity Follow-up Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED
- Version Label: marl_factorial_followup_v1

## Motivation and hypotheses

The completed MARL diagnostic found that independent actors and broadcast shop
context each improved the current parameter-shared MAPPO on all five training
seeds. Two questions remain before spending the reserved final test panel:

1. **Combination hypothesis**: independent actors with broadcast context reduce
   validation objective relative to the independent-actor incumbent without
   increasing failures.
2. **Factorial interaction question**: determine whether the joint benefit is
   additive, synergistic, or redundant using
   `combined - independent - broadcast + shared_no_context`.
3. **Capacity hypothesis**: if independent actors beat a shared actor with a
   matched actor-parameter count, the prior gain is not explained by actor
   capacity alone and is consistent with machine specialization.

These conclusions remain bounded to the current synthetic benchmark instance.

## Locked conditions

Four completed conditions are re-evaluated from frozen checkpoints:

- `centralized_shared_scorer_entropy`
- `parameter_shared_mappo_global_critic`
- `independent_actor_mappo_global_critic`
- `parameter_shared_mappo_broadcast_context`

Two new conditions are trained:

- `independent_actor_mappo_broadcast_context`: one 64-unit actor per machine,
  eight broadcast aggregate shop features, and the existing global critic.
- `matched_capacity_shared_mappo_global_critic`: one shared actor whose hidden
  width is selected deterministically to match the total parameter count of the
  two independent 64-unit base actors within 0.5%; no broadcast context.

For the current two-machine, 23-feature benchmark, the independent actors have
11,522 parameters in total and the matched shared actor uses hidden width 95
with 11,496 parameters, a relative gap of approximately 0.23%.

## Metrics, seeds, and stopping

- **Primary endpoint**: per-training-seed mean validation objective difference,
  combined treatment minus independent-actor incumbent.
- **Capacity contrast**: independent actor minus matched-capacity shared actor.
- **Factorial interaction**: reported for objective and all secondary metrics.
- **Secondary metrics**: makespan, tardiness, failures, preventive and
  corrective maintenance, total cost, conflict counts, and rejection rate.
- **Training seeds**: 10000, 11000, 12000, 13000, 14000.
- **Budget**: 500,000 joint environment interactions per new condition/seed.
- **Full validation**: fresh seeds 54000--54199.
- **Local smoke validation**: seeds 53900--53904, one training seed, 256 steps.
- **Reserved final test**: seeds 50000--50099 remain unopened.
- **Stopping rule**: fixed budget only; no early stopping, adaptive retry,
  validation checkpoint selection, or test evaluation.
- **Replication unit**: independent training seed; environment seeds are paired
  repeated measurements.

## Promotion gate

The combined policy is eligible for a separately authorized final test only if:

1. mean objective is below the independent-actor incumbent;
2. at least four of five objective deltas are nonpositive;
3. mean failures do not increase; and
4. the coordination audit passes.

This is an engineering gate, not a statistical-significance claim. Passing the
gate does not automatically open the reserved test panel.

## Artifact schema and integrity gates

Each timestamped run writes:

- `marl_factorial_manifest.json`
- `benchmark_config.json`
- `marl_factorial_episodes.csv` and `.partial.csv`
- `marl_factorial_coordination.csv` and `.partial.csv`
- `marl_factorial_summary.json`
- per new condition and seed: `policy.pt`, five recovery checkpoints,
  `training_progress.csv`, and `training_episodes.csv`
- the terminal log created by `tee`

Completion requires 6 conditions x 5 training seeds x 200 validation seeds =
6,000 unique episode rows, complete source checkpoints, reward/objective
identity, zero invalid or duplicate executions, the capacity-match tolerance,
and an unopened final test panel. Training and multi-model evaluation use
`tqdm` progress bars.

## Local smoke command

```bash
uv run ht-pdm-fjsp-marl-factorial \
  --profile smoke \
  --device cpu \
  --centralized-source-run lab_results/shared_action_ppo_20260920T123821Z \
  --ctde-source-run lab_results/ctde_mappo_20260921T153014Z \
  --diagnostic-source-run lab_results/marl_diagnostic_20260921T165536Z \
  --output-dir artifacts/marl_factorial_smoke
```

## Ubuntu and retrieval workflow

The handoff supplies one foreground Ubuntu command to fetch the committed
branch, verify all three source runs, execute the full profile with `tqdm`, and
tee a timestamped log. A second Mac command retrieves the complete remote
artifact tree from `bachng@100.111.83.52` with `rsync -avhP --partial`.
