# Minimal HT-PdM-FJSP benchmark

This repository contains a small, reproducible event-driven benchmark for joint
flexible-job-shop scheduling, Weibull machine degradation, preventive/corrective
maintenance, and heterogeneous technician assignment.

The model is deliberately minimal. It is a simulator validation target, not a
claim that the synthetic parameters represent a specific factory.

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
```

## Test

```bash
uv run pytest
```

## Run the benchmark

```bash
uv run ht-pdm-fjsp-benchmark \
  --config configs/minimal_benchmark.json \
  --output-dir artifacts/minimal_benchmark \
  --seeds 0:100
```

Outputs include episode-level CSV, policy summary JSON, event traces, a config
snapshot, and a reproducibility fingerprint.

Failure shocks are keyed by seed, job, operation, machine, and retry number, so
policy comparisons use common random numbers without depending on event order.

## Core assumptions

- Effective machine age increases by `alpha * processing_time * load_factor`.
- Failure follows the conditional Weibull hazard over that age increment.
- A failed operation restarts after corrective maintenance.
- Preventive and corrective maintenance occupy both the machine and one eligible
  technician.
- Technician-specific duration and restoration effectiveness create the shared
  resource coupling.

See `docs/minimal_benchmark_plan.md` for scope and validation boundaries.

## Gymnasium environment

The registered environment is `HTPdMFJSP-v0`. It exposes a fixed discrete
action catalog, a dynamic `action_mask`, structured observations, and dense
reward components whose completed-episode return equals the negative objective.

```python
import gymnasium as gym
import ht_pdm_fjsp

env = gym.make("HTPdMFJSP-v0")
observation, info = env.reset(seed=0)
feasible_actions = observation["action_mask"].nonzero()[0]
```

Run all five Gymnasium baselines on common random-number seeds:

```bash
uv run ht-pdm-fjsp-gym-benchmark \
  --config configs/minimal_benchmark.json \
  --output-dir artifacts/gym_baselines \
  --seeds 0:100
```

The advanced panel includes masked SPT, health-threshold maintenance,
joint risk/technician assignment, reactive CP-SAT, and rolling-horizon
lookahead. See `docs/gymnasium_and_advanced_baselines_plan.md` for the exact
environment contract and validation gates.

## MaskablePPO smoke test

```bash
uv run ht-pdm-fjsp-ppo \
  --profile smoke \
  --output-dir artifacts/ppo_smoke
```

Training and evaluation display `tqdm` progress and save the model, monitor
files, CSV training log, episode-level results, summary, and runtime manifest.
SB3 metric tables are suppressed on the terminal and retained in
`training_log/progress.csv`.
The complete local-to-Ubuntu workflow and `rsync` retrieval command are in
`docs/ppo_experiment_plan.md`.

For independent-training-seed replication:

```bash
uv run ht-pdm-fjsp-ppo-replicates \
  --profile smoke \
  --output-dir artifacts/ppo_replicates_smoke
```

The full profile trains five independent models, evaluates shared held-out
seeds, evaluates intermediate checkpoints on validation seeds, and writes
partial artifacts after every completed replicate. See
`docs/ppo_multiseed_replication_plan.md`.

## MARL architecture diagnostic

The diagnostic runner compares the completed centralized and MAPPO checkpoints
with matched-budget PS-IPPO, independent-actor MAPPO, and broadcast-context
MAPPO treatments. It isolates centralized-critic, parameter-sharing, and actor
information effects while keeping the final test panel closed.

```bash
uv run ht-pdm-fjsp-marl-diagnostic \
  --profile smoke \
  --device cpu \
  --centralized-source-run lab_results/shared_action_ppo_20260920T123821Z \
  --ctde-source-run lab_results/ctde_mappo_20260921T153014Z \
  --output-dir artifacts/marl_diagnostic_smoke
```

See `docs/marl_diagnostic_plan.md` for the locked hypotheses, metrics, seeds,
stopping rule, artifact schema, and Ubuntu workflow.

Measure whether feasible operation and technician overlaps form small dynamic
machine-agent subteams before implementing a graph-aware value decomposition:

```bash
uv run ht-pdm-fjsp-resource-graph \
  --profile smoke \
  --device cpu \
  --source-run lab_results/marl_budget_screen_iql_qmix_full_cpu_20260923T030641Z \
  --output-dir artifacts/resource_graph_smoke
```

The diagnostic replays the locked 300k IQL and QMIX checkpoints, records the
six-agent resource-conflict graph at every dispatch epoch, and keeps the future
test panel sealed. See `docs/resource_conflict_graph_diagnostic_plan.md`.

Attribute that graph topology to broad feasibility versus policy intent:

```bash
uv run ht-pdm-fjsp-environment-attribution \
  --profile smoke \
  --device cpu \
  --source-run lab_results/marl_budget_screen_iql_qmix_full_cpu_20260923T030641Z \
  --output-dir artifacts/environment_attribution_smoke
```

This read-only diagnostic compares all-feasible, production-only, and policy
top-2 intent graphs on identical trajectories. See
`docs/environment_attribution_diagnostic_plan.md`.

Test that policy-intent graph as soft centralized-training context in a
capacity-matched full-team QPLEX mixer:

```bash
uv run ht-pdm-fjsp-qplex-soft-graph \
  --profile smoke \
  --device cpu \
  --output-dir artifacts/qplex_soft_graph_smoke
```

The four-way ablation retains the exact feed-forward agent utility baseline and
compares QMIX against QPLEX with null, all-feasible, and policy-intent top-2
graph inputs. See `docs/qplex_soft_graph_ablation_plan.md`.

The factorial follow-up completes the actor-sharing by broadcast-context design
and adds a parameter-count-matched shared actor. Run it after the diagnostic
artifacts are available:

```bash
uv run ht-pdm-fjsp-marl-factorial \
  --profile smoke \
  --device cpu \
  --centralized-source-run lab_results/shared_action_ppo_20260920T123821Z \
  --ctde-source-run lab_results/ctde_mappo_20260921T153014Z \
  --diagnostic-source-run lab_results/marl_diagnostic_20260921T165536Z \
  --output-dir artifacts/marl_factorial_smoke
```

See `docs/marl_factorial_followup_plan.md` for the locked factorial and capacity
contrasts.
