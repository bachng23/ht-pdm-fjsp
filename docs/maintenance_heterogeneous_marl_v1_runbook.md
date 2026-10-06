# Heterogeneous technician allocation and cooperative machine PPO v1

Branch `codex/maintenance-heterogeneous-marl-v1`, parent `de85e82`.
Protocol: `docs/maintenance_heterogeneous_marl_v1_plan.md`.

## What this tests

Four machines share passive technicians. Three machines favor the same specialist;
the fourth favors the other. Favored service is shorter and restores more health.
Worker ownership is retained through completion. Homogeneous K2 is a paired
reference treatment; homogeneous K4 is the nominal guard. Removing fallback
compatibility edges is a separate sensitivity that also reduces feasible capacity.
Row mean duration/restoration are matched, not effective throughput.

Three equally parameterized learned arms compare full matching, learned machine
selection with fixed allocation, and cooperative machine agents. Full-width
models each have 72,258 parameters; initial weights, public observations,
training environments and step budgets match per training seed. Cooperative
agents simultaneously nominate WAIT/worker, and a public rotating coordinator
resolves collisions. PPO evaluates raw nominations before arbitration with a
joint likelihood and shared critic. This is an explicit cooperative formulation
with shared execution information, not communication-free execution.

The two primary contrasts are full matching vs fixed allocation (>=3% priced,
>=2% base cost reduction), and cooperative vs centralized full matching (>=2%
priced reduction). Both need >=8/10 paired training-seed wins, upper97.5% CI<0
and nominal/wait guards. All three economic-vs-rule comparisons are secondary.
The inference unit is training replicas, conditional on a fixed test panel;
profile generator support is finite. A cooperative PASS supports this tested
method, not mathematical necessity of MARL. Planner references are not optima.

## Local verification

- Full repository regression: 539 passed, one existing SB3/rich tqdm warning,
  197.37 seconds. Final machine-cost/worker accounting additions: all24 new
  targeted tests passed in10.16 seconds. Ruff F checks and diff whitespace pass.
- New tests check full-width1536-step rollouts for all three arms with smoke
  training seed155000, physical matching masks, raw nomination likelihood,
  finite gradients, completion restoration, scalar/batched forecasts,
  homogeneous allocation invariance, seed exclusions, dev selection, adjusted
  paired CI gates, all1080 smoke action/cost/checkpoint replays and failure logs.
- Final Mac CLI smoke: `maintenance_heterogeneous_marl_v1_smoke_cpu_20261006T145037Z`;
  status COMPLETED. Three models,768 train steps,192 training episodes,
  48 optimizer steps,34 development episodes,216 test episodes,1080 decisions,
  864 machine rows. All verdicts ENGINEERING_ONLY.
- All manifest hashes/counts, identical initial weight hashes and paired training
  configuration/shock streams verified independently. Smoke full artifact path:
  `/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_heterogeneous_marl_v1_smoke_cpu_20261006T145037Z`.
- No new full scientific test rollouts were run on the Mac. Full requires clean
  committed source. Only metadata/config snapshots are prepared before test;
  all30 selected checkpoints must be frozen before any full test rollout.

## Lab run contract

Run `python -m ht_pdm_fjsp.maintenance_heterogeneous_marl --profile full --device cpu`
in a clean new detached worktree from the pushed branch, pinned to the handoff
commit, with its own `uv sync --frozen` environment. Store results in
`$HOME/ht-pdm-fjsp/artifacts/<new UTC run ID>` so canonical rsync collects them.
The calling checkout may contain another experiment; do not switch its branch.
Use tqdm and pipefail+tee. Keep the lab terminal open; do not use tmux. No resume,
no silent overwrite, no SSH or transfer by the assistant.

Full:30 models,14,745,600 training steps,1,228,800 train episodes,
307,200 optimizer steps,63,200 dev episodes,144,000 test episodes,
2,592,000 decisions and576,000 machine rows. Detailed CSV logs can be several GB;
this is an estimate, not a measured full-run size. Manifest includes all file
hashes, protocol/commit/runtime/configs and exact budgets. Failures preserve
partial artifacts and mark FAILED. Only COMPLETED results support analysis.

After push, hand off exactly one compound lab run command and the canonical
`rsync -avhP --partial` Mac command documented in `docs/experiment_workflow.md`.
