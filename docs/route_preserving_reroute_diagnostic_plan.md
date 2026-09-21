# Route-preserving reroute diagnostic

## Question and hypothesis

The route-preserving residual changes only which production alternative is selected,
but its validation objective is unstable across independent PPO training seeds. This
read-only diagnostic asks whether the observed reroutes are locally beneficial after
holding the production/maintenance/advance decision fixed.

The diagnostic hypothesis is that harmful seeds either cross small shared-score
margins too aggressively or prefer alternatives with systematically worse machine
health, load, processing time, or slack. No model is trained and no new data split is
opened.

## Frozen data and estimand

- Reuse the five trained route-preserving models and their matching frozen shared
  models.
- Replay exactly validation seeds 47000--47199 already opened by the source run.
- Keep reserved test seeds 50000--50099 unopened.
- At every production reroute, deep-clone the simulator state. Force the residual
  choice in one branch and the shared choice in the other, then continue both with
  the same deterministic route-preserving policy.
- Report residual-minus-shared deltas for objective, failures, maintenance, cost,
  tardiness, and makespan. Lower objective delta is beneficial.

Reroute events within a model are repeated descriptive observations. The independent
replication unit remains the PPO training seed; event rows must not be treated as
independent samples for confirmatory inference.

## Integrity checks and stopping rule

The run stops with an error unless all of the following hold:

1. production/maintenance/advance kinds always match the shared model;
2. non-production actions are identical;
3. every selected action is feasible;
4. the residual policy's frozen internal base action equals the loaded shared model;
5. every replayed episode matches the source result table; and
6. every forced residual branch reproduces the actual episode's complete trajectory
   fingerprint (task schedule, completion times, final machine ages, and metrics).

The diagnostic is complete after all requested source validation cells have been
replayed once. It does not adapt seeds or open the future test panel.

## Artifacts

The timestamped output directory contains the copied benchmark configuration,
running/completed manifest, episode audit table, per-reroute counterfactual table,
per-training-seed summary, and JSON diagnostic summary. Partial episode and event
tables are rewritten after each completed training seed for recoverability. A single
`tqdm` bar reports episode progress.
