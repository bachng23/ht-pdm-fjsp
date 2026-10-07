# Maintenance solver comparison v1 — engineering handoff evidence

Date: 2026-10-07 Asia/Taipei. Status: ENGINEERING_ONLY; no full scientific
training or held-out test rollout was run locally. Parent commit693c348.
Branch: `codex/maintenance-solver-comparison-v1`.

## Final Mac smoke

Directory: `artifacts/maintenance_solver_comparison_v1_smoke_cpu_20261007T143401Z`.
Console log is alongside the directory with `.log` suffix. Earlier smoke
`20261007T142838Z` is retained; it predates actor-only serial inference and final
phase-timing bookkeeping. Neither smoke informs scientific performance claims.

Final manifest COMPLETED;5 models,1,280 physical training steps,320 training
episodes,80 optimizer steps,46 development episodes,264 test episodes,
1,320 test decision intervals and1,056 machine rows. All exact budgets match.
Each arm has5,811 total parameters at smoke hidden16; all start with identical
tensors. Every model changed parameters, passed sampled/reevaluated likelihood
and complete-return reconciliation, and passed checkpoint save/reload.

Read-only final-artifact verification checked72 hashes/sizes, all600 learned
actions and diagnostics against selected checkpoints, and406 request records.
The359 terminal-censored requests reconcile exactly with terminal pending counts;
this tiny-budget behavior is engineering evidence, not service-quality evidence.
All summary verdicts are ENGINEERING_ONLY. Smoke uses three full-training-pool
profiles and does not open the full held-out test panel. Source/protocol/registry
snapshot bytes match the implemented files at handoff preparation.

Protocol SHA256:
`1c99669573ad6f6ed8f81d52a3b555bd8fdb40f0054b301925474a5edb68fced`.

Tests cover local-information invariance, actor-only serial execution without
critic access, exact preservation of prior central heads, serial symmetry and
all209 feasible nominal matchings, teacher-forced sequence likelihood, physical
per-machine reward sums, training updates, profile/seed separation, request
right-censoring, t-interval constants, statistical units/gates, grid duplication,
FAILED manifest retention, immutable outputs, and integrated physics replay.
Final full-repository validation: `.venv/bin/python -m pytest -q` — **573 passed**
in182.45 seconds; one existing SB3/rich progress-bar experimental warning.
Dependency setup: `uv sync --frozen --extra dev`; scientific lock unchanged.

Latency artifacts include48 timed decisions/controller for each of11 smoke
controllers (three passes over16 shared development states), cache diagnostics,
and explicit logical telemetry/reservation/dispatch payload counts. Timings from
this engineering run are not deployment performance claims.

## Full-run scope and limitations

Five learned arms ×10 replicas, six nonlearning controls. Full budget:
24,576,000 training steps,2,048,000 training episodes,512,000 optimizer steps,
82,560 development episodes,286,720 test episodes,5,160,960 decision rows,
1,146,880 machine rows, plus request/latency logs. Expect multiple GB; the prior
heterogeneous run had approximately half as many test decision rows. Runtime
depends on CPU and serial inference; no precise ETA inferred from smoke.

This study compares centralized rules/planners, centralized PPO, parameter-shared
independent local PPO and cooperative local/full-information machine policies.
It permits any family to win. Reservations are serial communication; local
policies do not have communication-free execution. The local learning contrast
changes return/critic/ratio together, not critic alone. Labeled physical profiles
are disjoint, but permutation-equivalent configurations may occur across splits.
The environment remains maintenance dispatch, not a complete FJSP or plant model.

Run only through the human lab handoff per `docs/experiment_workflow.md`.
CPU only, Linux full profile, clean committed checkout, new UTC artifact path,
no resume, no SSH/remote launch/transfer by assistant. Do not close the lab
terminal while running. `uv sync --frozen` is sufficient; no imported checkpoint,
new third-party dependency, teacher dataset or external model service is needed.
