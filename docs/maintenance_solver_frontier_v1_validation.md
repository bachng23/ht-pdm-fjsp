# Maintenance solver frontier v1 — Mac validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation and engineering verification
- Origin Date: 2026-10-10 Asia/Taipei
- Verification Status: VERIFIED engineering smoke; UNVERIFIED full scientific result
- Version Label: maintenance_solver_frontier_v1_validation

## Verified behavior

The new runner reuses frozen centralized checkpoints and existing audited physical
simulation. It selects rule and planner continuations ONLY on development,
reports all planner budgets, measures isolated shared-state timing and actual
closed-loop timing, and records both quality and compute with a locked2% cost
margin. No new training, scaling or architecture change is included.

Fresh scenario-major planner streams give exactly nested forecasts across budgets.
All candidate rollout scores and selected matchings match independent scalar
transition rollouts on identical forecast events, including service completion,
heterogeneous skill eligibility and overdue waiting costs.

## Tests and packaging

- Full regression suite:630passed, one upstream tqdm/rich experimental warning,
  227.47seconds. No test opened the new full shock trajectories.
- New focused suite:20passed. Repeated after final plot tick/profile-label fix:
  20passed in13.95seconds. Covers source hash corruption, nonfinal selection,
  registered seeds, vectorized/scalar planner equivalence, crossed paired
  bootstrap, practical joint screens, duplicate/missing grid rejection, immutable
  output, failure artifacts, Mac full-run prohibition and integrated smoke replay.
- uv lock plus uv sync --frozen --extra dev succeeded. Matplotlib is explicitly
  declared and locked for scientific PNG/PDF exports; CLI entry is installed.
- Ruff0.14.0 F checks and git diff --check passed. Only new files and the project
  dependency/entry declarations are changed; prior benchmark runners unchanged.

## CLI smoke with real frozen checkpoint

Final run:artifacts/maintenance_solver_frontier_v1_smoke_cpu_20261010T055303Z.
COMPLETED, CPU1thread; source run maintenance_training_coverage_v1_full_cpu_
20261009T044236Z, covered central_matching_ppo seed180000. Source checkpoint
SHA256:750253918db537f05194fb76a3136def806729d65f912f45180a8c75d3ae2e57.
Full source preflight was separately applied to all10required historical models;
all identity/final-selection/relevant hash checks passed, without evaluating
new full shocks.

Smoke counts:0new training steps,1frozen model,64development episodes,
80test episodes,400physical decisions,320machine rows,40shared states,
400warm timing rows and40cold timing rows. tqdm bars appeared in the saved log.
Source scientific test profiles/shocks were not opened by this smoke; both
engineering physical profiles come from the historical FULL TRAIN pool.

Independent audit:
-66artifact hashes AND sizes match;1,907,046bytes total.
-400closed-loop actions regenerated,400physical transitions/state/cost records
  replayed exactly,400warm and40cold shared-state actions regenerated.
-Zero mismatches across all840action replays and400physical transitions.
-Source checkpoint copy hash matches the original.
-Final PNG/PDF visually inspected; non-overlapping axis labels and explicit
  SMOKE label. Plot points are engineering evidence only.

The smoke used a new timestamp after visual QA; the prior smoke directory was
preserved. Unit-test synthetic source fixtures are distinct from the real-source
CLI smoke and are never described as trained models or scientific evidence.

## Remaining human-only full run

Full run:10frozen replicas,1rule,3planner budgets,20,480development episodes,
71,680test episodes,1,290,240test physical ticks,286,720machine rows,
2,304common states,516,096warm timing rows and112cold observations.
No exact wall-time estimate is inferred from H4/H6 smoke. Artifacts are expected
in the hundreds of MB; provision at least a few GB rather than a tight quota.

Full Linux execution requires a clean committed checkout and explicit single-
thread environment. Source preflight fails before selection/evaluation if a
required checkpoint is absent/corrupt/wrong. Interrupts preserve partial data
and a FAILED manifest. No resume or silent overwrite. Assistant does not SSH,
launch the full experiment or rsync; human commands follow commit/push.

## Interpretation limits retained in output

Only quality/compute at current N4, familiar labeled profiles and fresh random
shocks. No workload or zero-shot scaling generalization, permanent-starvation
claim, optimal planner certificate or operational deadline. Bootstrap cost
intervals are approximate and multiplicity-adjusted nominally; waiting/pending
and latency screens remain descriptive, not a joint confirmatory test. RL family
mean is not an ensemble; historical training wall time is not zero and cannot be
converted into CPU amortization without measurements unavailable in source run.
