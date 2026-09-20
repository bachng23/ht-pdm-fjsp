# Gymnasium environment and advanced-baseline plan

## Material Passport

- Artifact: code experiment plan
- Status: implementation specification
- Parent benchmark: `ht_pdm_fjsp_minimal_v1`
- Primary validation unit: policy × stochastic seed
- Scope: environment and baseline validation, not algorithmic confirmation

## Environment contract

### Decision epochs

An agent repeatedly selects one feasible atomic action at the current event
time. It may fill multiple idle resources before advancing simulation time.

### Action catalog

1. `advance`: process the next simultaneous event set.
2. `production(job, operation, machine)`.
3. `preventive(machine, technician)`.
4. `corrective(machine, technician)`.

The catalog has fixed size for a configured instance. `action_mask` identifies
currently feasible choices. Invalid actions are handled safely with a penalty,
rather than crashing Gymnasium rollouts.

### Observation

A Gymnasium `Dict` containing:

- normalized time;
- per-job completion, activity, progress, and slack;
- per-machine status, effective-age ratio, Weibull shape, and PM readiness;
- technician availability and skill matrix;
- per-action type and operational features;
- binary action mask.

### Reward identity

Dense reward components charge elapsed makespan, final job tardiness,
preventive/corrective starts, maintenance downtime, maintenance waiting, and
invalid actions. For a completed episode:

`episode_return = -objective`.

## Baselines

1. `masked_spt`: corrective priority, shortest-processing-time production.
2. `health_threshold`: adds risk-threshold preventive maintenance.
3. `joint_risk_greedy`: compares expected failure consequence with
   technician-specific preventive cost/duration/restoration.
4. `cp_sat_reactive`: deterministic CP-SAT FJSP plan with reactive
   condition-based maintenance and corrective repair.
5. `rolling_horizon`: bounded lookahead over cloned environment states with a
   joint-risk-policy rollout for the tail value. Planning uses common scenario
   seeds that are independent of the evaluated episode seed, so the policy
   cannot inspect unrevealed future failures. A conservative improvement margin
   prevents deviations from the base policy when Monte Carlo gains are small.

## Validation gates

- Pass Gymnasium environment checker.
- Observation belongs to the declared space after every step.
- At least one action is feasible at every nonterminal state.
- Action mask blocks resource, skill, precedence, and status violations.
- Completed traces pass machine, technician, precedence, and eligibility audit.
- Reward identity holds for every baseline and seed.
- Same policy/seed produces identical trace, rewards, and metrics.
- CP-SAT returns a feasible production plan.
- Multi-seed smoke panel completes without truncation.
