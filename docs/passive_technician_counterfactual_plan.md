# Queue-aware counterfactual technician allocation

## Research question and hypothesis

Can a counterfactual credit-assignment mechanism that evaluates each feasible
technician choice against the same joint request state reduce objective cost and
queue waiting relative to MAPPO-CTDE on the corrected passive-technician cell?

The falsifiable hypothesis is that the proposed policy will have lower mean
objective and waiting on the common evaluation panel, without increasing
invalid requests or collapsing joint-action diversity.

## Algorithm intervention

The environment, action space, failure process, cost function, action masks and
FIFO resource semantics are unchanged. The baseline is centralized PPO with a
global actor/critic. The proposed learner uses decentralized local actors with
two structural components:

1. a technician-edge actor scores `defer` and each technician separately from
   the machine's resource-aware observation;
2. a centralized joint-action critic supplies a counterfactual advantage by
   replacing one machine's selected technician while holding all other requests
   fixed and averaging over that machine's masked policy.

FIFO is therefore an environment/reference semantics, not the proposed
algorithm. The proposed change is the technician-specific credit assignment.

## Locked protocol

- stress cell: three machines, two heterogeneous technicians, horizon 12;
- policies: `mappo_ctde` and `queue_aware_counterfactual`;
- train seeds: 11, 12, 13;
- evaluation seeds: 101--130;
- training budget: 5,000 episodes per training seed;
- smoke: train seed 11, evaluation seeds 101--103, 32 episodes;
- stopping: fixed episode budget; no checkpoint selection after evaluation;
- sealed test panel: not opened by this experiment.

## Metrics and gates

Primary metric: mean objective cost on the common evaluation panel.

Mechanism metrics: waiting, busy-technician requests, failures, jobs,
collisions, invalid requests, unique joint actions, defer fraction and request
count. The proposed method passes the engineering gate only if checkpoints
reload, all expected rows are present, objectives are nonnegative,
`invalid_requests=0`, and the sealed panel remains closed.

## Artifact schema

Each timestamped run contains `manifest.json`, `benchmark_config.json`,
`episodes.partial.csv`, `episodes.csv`, `coordination.csv`,
`training_progress.csv`, `summary.json`, and one serialized checkpoint per
policy and training seed. The manifest records protocol, device, source
revision, output files and completion status.

## Interpretation limits

This is a mechanism experiment, not a claim of state-of-the-art performance.
An improvement over MAPPO supports the proposed credit-assignment mechanism on
this environment; it does not establish superiority on other maintenance or
scheduling benchmarks. A later study must test larger cells, alternative
contention semantics and exact small-instance gaps.
