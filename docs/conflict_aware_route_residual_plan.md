# Conflict-aware route-preserving residual PPO

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED until the full lab run completes
- Version Label: `conflict_aware_route_residual_v1`

## Question and hypothesis

The route-preserving residual preserved every production/maintenance/advance
decision, but it did not improve mean validation objective. Its counterfactual
diagnostic found 228 reroutes: 194 changed both job and machine and had zero
objective effect, while all 34 consequential reroutes involved actions sharing a
machine or a job-operation.

This experiment tests whether restricting residual ranking to real production
conflicts reduces credit-assignment noise and improves training-seed stability.
The hypothesis is that conflict-aware ranking will retain the structural safety
of the route-preserving policy while lowering mean objective relative to both
the shared baseline and the unrestricted route-preserving residual.

## Locked intervention

The matching completed `shared_scorer_entropy` actor initializes and remains the
frozen base. The shared policy selects the route first. Advance, preventive, and
corrective actions are executed unchanged. If the shared action is production,
the residual may replace it only with a feasible production action that:

1. uses the same machine, or
2. represents the same job-operation on another eligible machine.

The shared action is always included in its own conflict set. PPO log
probabilities use the exact marginal distribution of the coupled base draw and
conflict-conditional residual draw. The residual head and critic may train; the
shared actor may not.

## Locked experiment

- Conditions: completed `shared_scorer_entropy`, completed unrestricted
  `route_preserving_residual_entropy`, and new
  `conflict_aware_route_preserving_residual_entropy`.
- Training seeds: 10000, 11000, 12000, 13000, 14000.
- Budget: 500,000 requested PPO steps per new model; four environments,
  `n_steps=1024`, batch size 256, 10 epochs, entropy coefficient 0.01.
- Fresh validation seeds: 48000--48199.
- Reserved future test seeds: 50000--50099 remain unopened.
- Independent replication unit: PPO training seed, not episode or decision.
- No early stopping, adaptive retry, checkpoint selection, or threshold sweep.

Primary endpoint: mean validation objective by training seed, conflict-aware
treatment minus shared baseline. Secondary comparisons include treatment minus
the unrestricted route-preserving residual, failures, maintenance, cost,
tardiness, makespan, across-seed dispersion, and reroute composition.

## Validity and decision rules

The run is invalid unless all action-mask and source-provenance checks pass and
the treatment has zero:

- decision-kind mismatches;
- non-production action mismatches;
- invalid actions; and
- production reroutes outside the base action's conflict set.

The architecture is eligible for later held-out testing only if its mean
objective is below the shared baseline, at least four of five training-seed
deltas are nonpositive, mean failures do not increase, and its across-seed
objective dispersion does not exceed the unrestricted residual. These are
predeclared engineering promotion criteria, not significance claims.

## Artifacts and stopping

The runner writes a timestamped directory containing the copied benchmark
configuration, manifest, three-condition episode table, per-seed route audits,
summary JSON, five treatment models, monitor logs, and partial tables after each
completed cell. Training and source evaluation expose `tqdm` progress. The run
stops after the fixed five-model budget and 200-seed validation panel; it never
opens the future test panel.
