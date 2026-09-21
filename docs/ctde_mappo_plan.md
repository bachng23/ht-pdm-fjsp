# Machine-Agent MAPPO-CTDE Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Objective**: replace the centralized single actor with genuine machine
  agents trained by centralized training and executed from local observations.
- **Hypothesis**: parameter-sharing MAPPO-CTDE reduces validation objective
  versus the completed centralized `shared_scorer_entropy` baseline without
  increasing failures.
- **Independent variable**: agent architecture only. The benchmark simulator,
  linear scalarized team reward, cost weights, training budget, and evaluation
  metrics remain fixed.
- **Type**: multi-agent reinforcement-learning training experiment.

## CTDE Architecture

- One agent per machine.
- A parameter-shared actor scores a static local action catalog containing wait,
  production alternatives for that machine, and its maintenance alternatives.
- Actor inputs contain only the local machine state, candidate job/action
  features, candidate technician availability, and current time.
- The actor never consumes another machine's state or the global state.
- During training only, one centralized critic consumes the complete shop state.
- The joint policy is the product of the machine-agent categorical policies;
  PPO uses the summed joint log probability and a shared team advantage.
- A deterministic round-robin machine resolver rejects duplicate job-operation
  claims and shared-technician conflicts before executing the accepted joint
  action, without permanently favoring one machine.
- Execution is decentralized: actors do not communicate and do not receive the
  critic or other agents' proposals.

## Locked Design

- **Treatment**: `machine_agents_mappo_ctde`.
- **Control**: completed centralized `shared_scorer_entropy` checkpoints.
- **Training seeds**: 10000, 11000, 12000, 13000, 14000.
- **Budget**: 500,000 joint environment steps per seed.
- **Smoke-only panel**: seeds 51900--51904.
- **Fresh full validation**: seeds 52000--52199, unopened until the code is
  tested, committed, and run on the lab.
- **Reserved final test**: seeds 50000--50099 remain unopened.
- **Primary endpoint**: mean objective per independent training seed, treatment
  minus control.
- **Secondary endpoints**: makespan, tardiness, failures, preventive and
  corrective maintenance, total cost, conflict and rejection rates.

## Promotion Rule

The treatment is eligible for the single future held-out test only if all are
true:

1. mean objective delta versus centralized control is negative;
2. at least four of five training-seed deltas are nonpositive;
3. mean failures do not increase; and
4. the coordination/architecture audit passes.

This is an engineering gate, not a statistical-significance claim.

## Required Audits

- Exactly one agent exists per configured machine.
- The shared actor accepts local tensors only; the centralized state is wired
  only to the value critic.
- No masked or simulator-invalid action is executed.
- Duplicate operation and technician claims are resolved and counted.
- Episode return equals negative benchmark objective.
- Every condition/seed/validation-seed cell exists exactly once.

## Artifacts

Each timestamped run writes:

- `ctde_manifest.json`
- `benchmark_config.json`
- `ctde_episodes.csv` and `.partial.csv`
- `ctde_coordination_audit.csv` and `.partial.csv`
- `ctde_summary.json`
- per-seed `mappo.pt`, five checkpoints, `training_progress.csv`, and
  `training_episodes.csv`
- the terminal log created by `tee`

Training and multi-seed loops display `tqdm` progress. A failed or interrupted
run remains separate and is never silently overwritten.
