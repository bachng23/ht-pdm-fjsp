# MARL Architecture Diagnostic Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED
- Version Label: marl_diagnostic_v1

## Research question and hypotheses

The completed parameter-shared MAPPO-CTDE run shortened makespan but increased
maintenance and tardiness enough to lose to the centralized PPO reference. This
experiment isolates three explanations while keeping the simulator, action
resolver, reward, PPO hyperparameters, training seeds, and interaction budget
fixed.

1. **Centralized-critic hypothesis**: if parameter-shared IPPO with a local
   critic matches or beats current MAPPO, the centralized critic is not adding
   useful credit-assignment information.
2. **Parameter-sharing hypothesis**: if independent machine actors beat the
   shared actor with the same centralized critic, actor sharing is preventing
   machine specialization.
3. **Information-bottleneck hypothesis**: if a shared actor augmented with
   observable aggregate shop context beats current MAPPO, decentralized actor
   information is the main limitation.

These are diagnostic contrasts, not independent claims that any treatment is a
new state of the art.

## Locked conditions

| Condition | Actor | Critic | Training |
|---|---|---|---|
| `centralized_shared_scorer_entropy` | completed global shared-action actor | global | reused checkpoint |
| `parameter_shared_mappo_global_critic` | completed shared local actor | global | reused checkpoint |
| `parameter_shared_ippo_local_critic` | shared local actor | shared local critic | new |
| `independent_actor_mappo_global_critic` | one actor network per machine | global | new |
| `parameter_shared_mappo_broadcast_context` | shared local actor plus eight shop aggregates | global | new |

The broadcast vector contains completed-job fraction, in-process-job fraction,
mean job progress, minimum due-date slack, other-machine idle fraction, mean and
maximum other-machine normalized age, and idle-technician fraction. It exposes
aggregates only, not another machine's identity or action proposal.

## Metrics, seeds, and stopping rule

- **Primary endpoint**: mean validation objective within each independent
  training seed; each diagnostic treatment minus current MAPPO.
- **Secondary endpoints**: makespan, tardiness, failures, preventive and
  corrective maintenance, total cost, conflict counts, and rejection rate.
- **Training seeds**: 10000, 11000, 12000, 13000, 14000.
- **Budget**: 500,000 joint environment interactions for every newly trained
  condition and seed.
- **Full validation**: seeds 53000--53199, unused by previous repository runs.
- **Local smoke panel**: seeds 52900--52904 with one training seed and 256 steps.
- **Reserved final test**: seeds 50000--50099 remain unopened.
- **Stopping**: fixed interaction budget. No early stopping, adaptive retry,
  validation-based checkpoint selection, or test evaluation.
- **Replication unit**: independent training seed. Validation episodes are
  paired repeated measurements, not independent training replicates.

## Interpretation matrix

- MAPPO approximately equal to IPPO and both worse than centralized PPO:
  centralized critic does not overcome the execution-time information bottleneck.
- MAPPO better than IPPO: the centralized critic contributes useful training
  information.
- Independent-actor MAPPO better than current MAPPO: parameter sharing is a
  limiting assumption.
- Broadcast-context MAPPO better than current MAPPO: actor observation is a
  limiting assumption.
- Every MARL condition shows shorter makespan but higher maintenance cost:
  inspect reward/action formulation before adding another optimizer.

The comparisons diagnose associations under controlled interventions; they do
not by themselves prove a unique causal mechanism on other benchmark instances.

## Artifact schema and integrity gates

Every run uses a new timestamped directory and writes:

- `marl_diagnostic_manifest.json`
- `benchmark_config.json`
- `marl_diagnostic_episodes.csv` and `.partial.csv`
- `marl_diagnostic_coordination.csv` and `.partial.csv`
- `marl_diagnostic_summary.json`
- for each new condition and seed: `policy.pt`, five recovery checkpoints,
  `training_progress.csv`, and `training_episodes.csv`
- the terminal log created by `tee`

Completion requires exactly 5 conditions x 5 training seeds x 200 validation
seeds = 5,000 unique episode rows, reward/objective identity, zero invalid or
duplicate executions, complete source checkpoints, and a closed final-test panel.
Training and multi-model loops expose `tqdm` progress.

## Local smoke command

```bash
uv run ht-pdm-fjsp-marl-diagnostic \
  --profile smoke \
  --device cpu \
  --centralized-source-run lab_results/shared_action_ppo_20260920T123821Z \
  --ctde-source-run lab_results/ctde_mappo_20260921T153014Z \
  --output-dir artifacts/marl_diagnostic_smoke
```

## Lab and retrieval workflow

The handoff command will fetch the committed branch, verify both source runs,
run the full profile on Ubuntu with `tqdm`, and tee a timestamped log. The Mac
retrieval command uses `rsync -avhP --partial` to preserve partial transfers and
retrieve the complete remote artifact tree from `bachng@100.111.83.52`.
