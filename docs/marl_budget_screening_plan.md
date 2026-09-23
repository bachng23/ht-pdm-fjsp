# MARL budget-screening protocol

## Purpose and hypothesis

This is an engineering screen, not a confirmatory algorithm comparison. It keeps
the locked Brandimarte MK01-derived `two_specialists_x2_0` environment with six
machine agents. The hypothesis is that batched collection from eight independent
environments removes enough Python/inference overhead that IQL and QMIX can be
screened cheaply, and that their useful learning may already plateau by 200k or
300k joint environment transitions.

## Design

- Algorithms: cooperative IQL, QMIX, and independent-actor MAPPO with global
  critic.
- One development training seed per algorithm: `76000`.
- Checkpoints: 100k, 200k, 300k, 400k, and 500k transitions.
- Paired development evaluation panel: seeds `61900:61950` (50 episodes).
- Collection: eight independent environments. IQL/QMIX use one batched action
  inference call per vector step and retain one gradient update per four stored
  transitions. MAPPO uses 500-step rollouts so every checkpoint is exact.
- The future test panel `62000:62100` remains sealed.

## Outcomes and stopping rule

The primary endpoint is mean objective at each checkpoint. Secondary endpoints
are makespan, maintenance wait, failures, three-way steps, and three-way episode
incidence. Training always stops at 500k during this screen; there is no
data-dependent early stop.

For each algorithm, select the earliest of 200k or 300k that simultaneously:

1. has mean objective within 5% of the best mean objective observed from
   100k--500k;
2. has three-way episode incidence within 0.02 absolute of its 500k value; and
3. has mean failures no more than 0.10 above the best checkpoint.

The proposed budget for a later replicated run is the maximum selected budget
across the three algorithms. If any algorithm fails the rule, do not claim a
200k/300k plateau; inspect the curves before spending on replicated training.

## Artifact contract

Each timestamped run contains the raw and scaled configs, a manifest with exact
software/settings hashes, five checkpoints per algorithm, training progress and
episode logs, checkpoint-level episode/decision/coordination CSV files, and a
JSON summary containing the locked selection rule and gate result. Partial CSVs
are refreshed after every evaluated checkpoint. Coordination audits require zero
invalid, duplicate-operation, and duplicate-technician executions.

The `smoke` profile uses 400 transitions, four environments, and three evaluation
episodes only to validate execution and artifact shape. Its output must not be
interpreted as a budget recommendation.
