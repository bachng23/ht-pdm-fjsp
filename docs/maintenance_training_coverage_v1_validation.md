# Maintenance training coverage v1 — macOS validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized implementation, engineering validation
- Origin Date: 2026-10-09 Asia/Taipei
- Verification Status: VERIFIED for local engineering gates only
- Version Label: maintenance_training_coverage_v1
- Scientific full result: NOT RUN; human lab handoff pending

Branch: codex/maintenance-training-coverage-v1. Parent09e325f.
Protocol: docs/maintenance_training_coverage_v1_plan.md.

## Verification

- `uv lock --offline` and frozen dependency sync succeeded; no lockfile change.
- `uv run --offline --extra dev pytest -q`:595 passed,1 upstream progress-bar warning,
  189.38 seconds. Includes regression of all inherited modules and10 coverage tests.
- Code formatter preserved both new files' syntax trees exactly. Final panel-open
  metadata boundary was moved before test execution so failures also report opening.
- Final focused rerun:10 passed in4.28 seconds.
- Cached ruff0.14.0 undefined/unused-name checks: PASS.
- Final CLI smoke: artifacts/maintenance_training_coverage_v1_smoke_cpu_20261009T041100Z,
  process exit0, manifestCOMPLETED. Its76 artifact hashes independently checked.
- tqdm displayed in training, development, latency and multi-seed evaluation.

Exact smoke counts:6 models,1536 training ticks,384 training episodes,96 optimizer
steps,24 checkpoint files,40 development episodes,128 test episodes,640 evaluation
intervals,512 actual machine rows and384 common-state latency samples.
Per model legacy52 specialized+12 nominal; covered26 specialized+26 skill-mask+
12 nominal. Smoke64 episodes is not an integer number of five-episode schedule
cycles; full40960 episodes is exact. All six cells share initial tensor digest and
training configuration/shock sequence digest. Parameters changed and every
checkpoint reload matched tensors plus actions/probabilities/values. All evaluation
requests reconcile served/censored waiting cost with episode metrics exactly.

Tests independently replay smoke logged physical transitions, verify condition
schedule changes only specified positions, verify nominal configurations/shocks
stay paired, distinguish within-policy gain from difference-in-differences,
reject missing/duplicate grids, reject full execution on Mac, enforce preservation
guards, record failed manifests/partial files and refuse existing nonempty outputs.
No full fresh shock trajectories were opened by engineering smoke/tests.

## Handoff scope

Full:60 models x491520 ticks =29491200 training ticks,2457600 training episodes,
614400 optimizer steps,360 checkpoint files,40960 development episodes,
317440 test episodes,5713920 evaluation intervals,1269760 machine rows and8928
common-state latency samples. Runs from fresh initialization; no downloaded
checkpoint dependency. Full scientific verdict remains pending.

Prepare several GB for per-training-episode, test/request/machine CSVs and weights;
actual size depends on request counts. No full decisions.csv, reducing artifact
volume. No resume; each invocation creates a new UTC timestamp directory. Keep
lab terminal open. The assistant must not SSH/run lab or initiate rsync.
