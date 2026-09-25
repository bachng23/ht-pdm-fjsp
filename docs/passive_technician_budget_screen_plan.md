# Passive technician training-budget screen

## Research question and falsifiable hypothesis

Does the poor performance of the current learned policies come mainly from an
insufficient training budget? The falsifiable hypothesis is that increasing the
fixed training budget from 5,000 to 10,000 and 20,000 episodes will reduce mean
objective cost and training-seed variance on the same evaluation panel. If the
curves plateau or worsen, training duration alone is not a sufficient
explanation and the remaining limitation is likely algorithmic or
representational.

## Locked protocol

The experiment uses the existing passive shared-technician simulator and its
three-machine/two-technician FIFO stress cell. Technicians remain passive,
heterogeneous resources. The compared learned policies are independent
tabular Q-learning, independent PPO, and centralized PPO. Fixed random-feasible
and skill-aware FIFO dispatchers are included as non-learning reference
policies with `budget=0`.

The primary metric is mean total objective cost. Secondary metrics are
failures, maintenance jobs, queue waiting, and proposal collisions. Every
budget for every learned policy is evaluated on the same evaluation seeds, so
budget differences are paired. Training seeds are 11, 12, and 13; evaluation
seeds are 101--200. No sealed test seeds are opened. The smoke profile uses
training seed 11, evaluation seeds 101--103, and budgets 8, 16, and 32.

Training uses a fixed budget with no adaptive stopping. Each model is trained
once through the largest budget (20,000 in full runs) and serialized at each
locked checkpoint. A budget is not selected after inspecting the results.

## Smoke and full-run gates

The smoke run must complete all expected rows, save and reload every model
checkpoint, write the manifest/CSV/JSON artifact schema, record `train_seed` on
all learned-policy rows, show training and evaluation progress, and keep the
sealed test panel closed. A full run is complete only when the manifest is
`COMPLETED`, all expected checkpoints and evaluation rows exist, and the same
audit gates pass.

## Artifact schema

Each timestamped output directory contains `manifest.json`,
`benchmark_config.json`, `training_progress.csv`, `episodes.partial.csv`,
`episodes.csv`, `coordination.csv`, `budget_summary.csv`, and `summary.json`.
Checkpoints are stored under
`<algorithm>/train_seed_<seed>/budget_<episodes>/model.(json|pt)`. The episode
table includes `policy`, `budget`, `seed`, `train_seed`, `objective`,
`failures`, `jobs`, `collisions`, and `waiting`. Fixed policies use `budget=0`
and an empty `train_seed`.

## Interpretation limits

This is a training-budget diagnostic, not a claim that more episodes produce a
better algorithm. Results are confirmatory only for the locked stress cell and
seed panels. A curve that improves with budget supports the undertraining
hypothesis but does not identify whether PPO optimization, observation design,
or credit assignment is the cause. The exact dynamic-programming oracle remains
limited to the small two-machine cell and is not used as an exact optimum for
the stress cell.
