# Objective and observation consistency fix

## Hypothesis

The previous full run mixed training objectives and exposed too little resource
information to the learning policies. Adding a positive FIFO waiting cost,
removing the Q-learning-only collision surcharge, and exposing service-time and
queue features should make policy comparisons fairer and make technician
selection observable to the agents.

## Changes under test

- Add a locked `queue_waiting_cost` to the passive-technician objective.
- Charge FIFO waiting consistently through the environment reward.
- Train every baseline on the exact environment cost; do not add a
  Q-learning-only collision term.
- Replace the scalar encoded observation with a structured per-machine vector
  containing time, age, failure state, technician availability, remaining busy
  time, queue length, queue position, and machine-technician service time.
- Evaluate the exact oracle and every baseline with the same configuration.

## Metrics

Primary: total objective cost.

Secondary: failures, maintenance jobs, collisions, queue waiting, downtime,
and technician utilization diagnostics.

## Seed split and stopping rule

The local smoke run uses one training seed and three evaluation seeds. The full
comparison retains training seeds 11--20 and evaluation seeds 101--200. The
episode budget remains fixed; no result-dependent extension is allowed.

## Artifact schema

The existing timestamped artifact schema is preserved. `benchmark_config.json`
records the new queue cost and observation version; `episodes.csv` records
policy, evaluation seed, training seed, objective, failures, jobs, collisions,
and waiting. Prior artifacts are never overwritten.
