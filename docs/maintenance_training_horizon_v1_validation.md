# Maintenance training horizon v1 — macOS engineering validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: user-authorized implementation and engineering validation
- Origin Date: 2026-10-10 Asia/Taipei
- Verification Status: VERIFIED for local engineering gates only
- Version Label: maintenance_training_horizon_v1
- Scientific full result: NOT RUN; human lab handoff pending

Branch codex/maintenance-training-horizon-v1; parent53d180a.
Protocol docs/maintenance_training_horizon_v1_plan.md.
No new architecture, optimizer or reward; both use frozen covered mixture.

## Verification evidence

- Frozen dependency sync and offline lock resolution succeeded; no lock change.
- Earlier focused run, horizon plus inherited coverage:22passed in13.15seconds.
- Final full regression suite:610passed,1upstream tqdm/rich warning in203.53seconds.
- Cached ruff0.14.0 F checks and git diff--check: PASS.
- CLI smoke artifacts/maintenance_training_horizon_v1_smoke_cpu_20261010T045339Z:
  exit0, manifestCOMPLETED, all78hashes and sizes checked independently.
- Independent replay of all768CLI smoke physical transitions matched logged
  following states and physical costs/metrics.
- tqdm observed in model training and multi-seed evaluation, with actual training
  horizonH4/H8 shown during smoke.

Actual CLI smoke counts:6models,1536training ticks,288training episodes,
96optimizer steps,24checkpoint files,40development episodes,128test episodes,
768evaluation intervals,512machine rows,1152latency samples on both eval horizons.
Smoke h12/h24 labels use actual4/8 respectively; each model256physical ticks and
16optimizer steps. Short branch64episodes=26specialized+26skill-mask+12nominal;
long32episodes=13specialized+13skill-mask+6nominal. Full mixture percentages
are exact40/40/20; smoke cycle remainder is explicitly recorded.

Within each regime all algorithms shared full episode seed digest; all six
cells shared first32episode seed digest and initial tensor digest. Raw CLI
training rows independently confirmed prefix identities/condition pairing.
Final-only selection; weights changed, save/load tensors/actions/probabilities
and values audited, actual training horizons in checkpoint and CSV identity.
Runtime checks reconcile every request's served/censored overdue cost with
its episode and each machine's charges with team cost.

A dedicated unit test also trained tiny models at ACTUAL12/24 horizons with
1536physical ticks and8optimizer updates each (one rollout, training-pool
profiles only), checking128/64complete episodes, Monte Carlo return boundaries,
PPO teacher-forced probabilities and common64episode seed prefix. No held-out
full test trajectories opened. This is engineering, not full scientific evidence.

Tests cover fresh seed registry, same configuration excluding horizon and event
prefixes, unequal episode counts under equal tick/optimizer budgets, checkpoint
contract rejection, difference-in-differences, missing/duplicate evaluation grid,
all homogeneous guards plus specialized/nominal cost guards, nonpositive gap
NOT_APPLICABLE, failed-run partial files and actual protocol identity, immutable
output directories, Mac full-run prohibition and inherited coverage regression.

## Full handoff scope

60models,29491200training ticks,1843200training episodes,614400optimizer steps;
360checkpoint files,40960development episodes,317440test episodes,
5713920evaluation intervals,1269760machine rows,26784common-state latency samples.
Reserve several GB for CSVs, snapshots and weights; previous coverage run was
about1.39GB, actual horizon-run size varies. Similar fixed compute scope to the
previous roughly20hour run; wall time is not guaranteed. CPU,one torch thread,
no downloaded checkpoint prerequisite. No resume or silent overwrite.

Assistant must not SSH/run the lab or initiate rsync. User executes the two
canonical handoff commands; keep terminal open. Fresh full test seeds195000..
195019 remain sealed until the human full run, while familiar profile labels
are reused explicitly. Waiting guard is relative preservation, not absolute SLA.
