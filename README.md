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
The complete local-to-Ubuntu workflow and `rsync` retrieval command are in
`docs/ppo_experiment_plan.md`.
