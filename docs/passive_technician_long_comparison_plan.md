# Long comparison for Technician-aware QMIX

## Research question

Does technician-aware queue-conditioned value decomposition outperform
Independent Q, standard value decomposition and actor-critic references when
all methods receive a substantially longer training budget?

## Candidate algorithms

- `independent_q`: current strong tabular value baseline;
- `vdn`: shared neural value decomposition;
- `qmix`: monotonic centralized value decomposition;
- `tqmix`: proposed technician-edge local Q plus queue-conditioned mixer and
  counterfactual technician consistency loss;
- `mappo_ctde`: actor-critic reference;
- `queue_aware_counterfactual`: the previous actor-critic mechanism candidate.

The FIFO queue remains environment/reference semantics. It is not part of the
algorithm contribution.

## Locked protocol

- stress cell: three machines, two heterogeneous technicians, horizon 12;
- train seeds: 11, 12, 13;
- evaluation seeds: 101--200;
- value-learning budgets: 5,000, 20,000 and 50,000 episodes per seed;
- actor-critic reference budget: 50,000 episodes per seed;
- fixed policies evaluated on the same 100-seed panel;
- no sealed test evaluation;
- fixed-budget stopping, with no checkpoint selection after seeing evaluation.

The actor references are evaluated at the final long budget. Value-based
methods additionally produce learning curves at all three checkpoints so that
undertraining and late collapse can be separated from algorithm quality.

## T-QMIX intervention

T-QMIX uses a local Q-network with separate defer and technician-edge scores.
Its mixer receives the flattened global observation plus explicit technician
availability, busy-time and queue-length context. A counterfactual consistency
loss aligns local technician-value differences with the joint-value difference
obtained by replacing one machine's technician action while holding the other
requests fixed.

## Metrics and promotion gate

Primary metric: mean objective cost on the common evaluation panel.

Mechanism metrics: failures, service jobs, waiting, collisions, invalid
requests, busy-technician requests, unique joint actions, defer fraction and
request count. T-QMIX is promoted only if it beats Independent Q at the long
budget with acceptable across-seed variation, preserves feasibility, and does
not rely on joint-action collapse. Beating MAPPO alone is insufficient.

## Artifact schema

Each timestamped run contains `manifest.json`, `benchmark_config.json`,
`episodes.partial.csv`, `episodes.csv`, `coordination.csv`,
`training_progress.csv`, `budget_summary.csv`, `summary.json`, and serialized
checkpoints. The manifest records the different checkpoint protocol for value
curves versus actor references.

## Interpretation limits

This is a development comparison. A T-QMIX win on this cell does not establish
general superiority. Follow-up validation must use a larger contention panel,
skill-overlap variants and exact small-instance checks before a publication
claim.
