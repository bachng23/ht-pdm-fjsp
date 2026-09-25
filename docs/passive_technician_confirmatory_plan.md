# Confirmatory passive-technician value-decomposition experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-26
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Research question and hypotheses

The exploratory component ablation at Git revision
`5b147feb8e021f02888f89cac78fb700681f2bcd` found that the
counterfactual-consistency variant had the lowest final objective, while the
full T-QMIX combination did not outperform the single-component variants. This
experiment is a new confirmatory protocol and does not modify the exploratory
run.

Primary hypothesis H1: at 50,000 training episodes on the in-distribution
panel, `qmix_counterfactual` has a lower mean objective than standard `qmix`
across paired training seeds. H1 is supported only when the paired mean
difference is negative, its deterministic 95% training-seed bootstrap interval
is entirely below zero, and the exact two-sided sign-flip p-value is below
0.05.

Secondary hypotheses:

- the counterfactual advantage has the same direction in at least three of the
  four reporting scenarios;
- `qmix_queue` improves early sample efficiency at 20,000 episodes;
- full `tqmix` does not receive a promotion claim unless it improves over both
  standard QMIX and the corresponding single-component variants;
- every learned policy remains feasible and avoids joint-action collapse.

## Locked conditions

The four trained variants are `qmix`, `qmix_queue`,
`qmix_counterfactual`, and `tqmix`. `random_feasible` and
`skill_aware_fifo` are fixed references.

All models train only on the existing base stress configuration. Each saved
checkpoint is evaluated without adaptation on these reporting panels:

| Scenario | Horizon | Failure age / probability | Service times |
|---|---:|---:|---|
| `in_distribution` | 12 | 5 / 0.45 | `((2,4),(4,2),(3,3))` |
| `early_failure` | 18 | 4 / 0.60 | base |
| `slow_service` | 18 | 5 / 0.45 | `((3,5),(5,3),(4,4))` |
| `combined_pressure` | 18 | 4 / 0.60 | slow-service |

Costs, machine count, technician count, FIFO semantics, observation schema,
masked action space, optimizer, replay settings, and architecture definitions
remain fixed across conditions.

## Seed panels and stopping rule

- paired training seeds: 21--30;
- reporting evaluation seeds: 1001--1100;
- sealed test seeds: 2001--2100, recorded but not evaluated;
- checkpoints: exactly 20,000 and 50,000 training episodes;
- stopping: fixed budget, with no early stopping and no checkpoint or
  architecture selection on reporting results.

The independent statistical unit is the training seed. Evaluation seeds are
common random numbers used to estimate each trained policy's conditional mean;
they are not treated as independent training replicates.

## Metrics and analysis

Primary metric: objective cost, first averaged over the common reporting seeds
within each trained model, then compared as paired differences across the ten
training seeds.

Mechanism metrics: failures, maintenance jobs, queue waiting, collisions,
busy-technician requests, invalid requests, unique joint actions, and defer
fraction.

For every variant-versus-QMIX comparison the runner records the paired mean and
standard deviation, deterministic 95% bootstrap interval, exact sign-flip
p-value, seed wins/ties, and Holm-adjusted p-value within each
scenario-by-budget family. The prespecified H1 uses its raw exact p-value
because it is the single primary comparison; Holm values are secondary-family
diagnostics.

## Smoke and full gates

Smoke uses one training seed, three reporting seeds, and budgets 8 and 16. It
passes when the process exits normally, all 120 expected episode rows and eight
checkpoints exist, checkpoints reload, schemas are valid, progress files are
nonempty, feasibility audits pass, and the sealed panel remains closed.

The full engineering gate requires all 32,800 expected episode rows, all 80
checkpoints, complete algorithm/seed/budget/scenario coverage, finite
nonnegative objectives, zero invalid requests, non-collapsed final policies,
and a closed sealed panel. Scientific hypotheses may be false without making
the run an engineering failure.

## Artifact schema

Each timestamped run writes:

- `manifest.json`, `benchmark_config.json`, and `resolved_config.json`;
- `training_progress.partial.csv` and `training_progress.csv`;
- `episodes.partial.csv`, `episodes.csv`, and `coordination.csv`;
- `seed_summary.csv`, `scenario_summary.csv`, and `paired_effects.csv`;
- `summary.json`;
- `<algorithm>/train_seed_<seed>/budget_<budget>/model.pt`.

The manifest records Git revision and dirty state, runtime device, all seed
panels, scenario definitions, completion state, checkpoint and row counts, and
the engineering hard gate.

## Interpretation boundary

This experiment estimates algorithm effects inside the locked simulator and
four specified scenario panels. It does not establish performance on real
maintenance systems or on untested technician-resource structures. Reporting
scenario results are robustness evidence, not a replacement for the still
sealed final test panel.
