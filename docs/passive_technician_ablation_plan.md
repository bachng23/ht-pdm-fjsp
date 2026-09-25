# Technician-aware QMIX component ablation

## Research question

Which components are responsible for the T-QMIX improvement: technician-edge
local Q values, queue-conditioned mixing, or counterfactual consistency?

## Locked variants

| Variant | Edge Q head | Queue mixer | Counterfactual loss |
|---|---:|---:|---:|
| `qmix` | No | No | No |
| `qmix_edge` | Yes | No | No |
| `qmix_queue` | No | Yes | No |
| `qmix_counterfactual` | No | No | Yes |
| `tqmix` | Yes | Yes | Yes |

All variants use the same masked action space, FIFO resource semantics,
replay/training settings, seeds and evaluation panel. `Skill-aware FIFO` is a
fixed reference and is not trained.

## Protocol

- train seeds: 11, 12, 13;
- evaluation seeds: 101--200;
- budgets: 20,000 and 50,000 episodes per seed;
- fixed references: `random_feasible`, `skill_aware_fifo`;
- sealed test panel remains closed;
- fixed-budget stopping with no checkpoint selection after evaluation.

## Primary and mechanism metrics

Primary metric: mean objective cost on the common evaluation panel.

Mechanism metrics: failures, jobs, waiting, collisions, busy-technician
requests, invalid requests, unique joint actions and defer fraction.

The full T-QMIX promotion gate is lower final objective than standard QMIX,
lower or equal seed variation, no feasibility violations, and no joint-action
collapse. Component claims require comparison against the corresponding
single-component variants, not just against standard QMIX.

## Artifact contract

Each run writes `manifest.json`, `benchmark_config.json`,
`episodes.partial.csv`, `episodes.csv`, `coordination.csv`,
`training_progress.csv`, `budget_summary.csv`, `summary.json` and serialized
checkpoints for every variant, seed and budget.
