## Material Passport

- Type: implementation and local macOS engineering verification.
- Protocol: `maintenance_priority_allocation_v1`.
- Status: VERIFIED engineering smoke; full scientific result unavailable.
- Locked design and criteria: [plan](maintenance_priority_allocation_plan.md).
- Historical rationale: [cause-based design](passive_technician_cause_based_redesign_2026-10-05.md).
- Branch: `codex/maintenance-priority-allocation`; implementation base `9cff694b3b19f45c4219adf7728ebb5d59900c88`.

## Implemented behavior

New standalone pending-demand environment uses only immediate feasible matchings of machines to idle compatible passive technicians. Failed demands persist; optional PM candidates are reconsidered at each time. Services stop operating age/hazard, restore the machine at completion, and charge one unavailability cost through waiting/service. No FIFO, assignment teacher, collision penalty or forced-repair action is used.

An entity/edge PPO actor selects a machine or STOP, then a technician, reserves both and repeats. Selected pairs start simultaneously. The critic estimates team return; there is no QMIX mixer. The policy requires a common dispatch view and coordinated execution. Three equal-capacity arms learn both decisions, fix priority, or fix allocation. This study tests the decisions under the new MDP; it does not establish why old QMIX/FIFO failed.

Exact N2 diagnostics decompose the deployed policy gap into machine-set selection and technician allocation regret, weighted by that policy's state occupancy. They use optimal future continuation and supply no training labels. Runtime verifies the additive identity. Per-machine wait, deferral, terminal status, service starts by technician, and resource busy ticks are recorded; early versus terminal deferral remains descriptive.

## Verification

Dependencies installed using `uv sync --frozen --extra dev`. Complete repository suite: **405 passed**, one existing SB3 rich/tqdm warning, 189.12 s. After final formatting/metadata changes, the **36 new tests passed**, 4.79 s. Ruff 0.14.0 lint and format passed on the six new Python files.

Tests cover failed waiting, PM/CM occupancy, one/multiple interval completion, partial restoration, no aging/failure in service, terminal unfinished work, invalid/busy/sparse assignments, simultaneous action order, entity relabeling, hidden policy-independent shocks, padded variable-size actor batches, sample/replay probability agreement, finite gradients, unused control heads, complete-episode returns, confirmatory criteria, save/load, refusal to overwrite, preservation of FAILED runs, and disappearance of source files after startup. Independently known small examples distinguish STOP selection regret from allocation-only regret.

Full evaluation's small H5 graph preflight: **1,475 states**, 9 uniformly weighted initial states, Bellman residual **0**, below the locked 25,000-node cap. This is a dynamics/evaluator check, not a full training run or policy result.

Actual CLI smoke: `artifacts/maintenance_priority_allocation_smoke_cpu_20261005T035110Z`, stored on the local Mac repository; adjacent `.log` retains tqdm output. Invocation used `.venv/bin/ht-pdm-fjsp-maintenance-priority-allocation --profile smoke --device cpu --output-dir <fresh-run-directory>` from the isolated worktree.

| Artifact/budget | Verified count |
|---|---:|
| Saved/reloaded models | 3 |
| Physical training steps / optimizer steps | 144 / 24 |
| Training episodes / rollout logs | 36 / 6 |
| Test / development / rule-reference episodes | 63 / 3 / 22 |
| Test decision traces / per-machine rows | 243 / 243 |
| Exact policy rows | 4 |
| Engineering audits | 22 / 22 PASS |

Independent post-run audit replayed every test trace (63 episodes), checked recorded states/actions/cost components, counted all CSV rows, independently recomputed unavailability charges, checked exact regret identities, and compared every frozen source hash against the final source. Nonempty-output rejection and source-disappearance regression also passed in integration tests. Sealed panels were not evaluated.

Environment: macOS-27.0.1 arm64, Python 3.12.13, torch 2.14.0 CPU, numpy 2.5.3, tqdm 4.70.1; manifest elapsed 2.527 s. Source bundle SHA256: `8382b3607bc70703a928600267d7602ae3d6af4e7acfc7931d905535c7e5c4a5`. Smoke manifest records the base commit with `git_dirty=true` because verification precedes the required commit; frozen source identifies the tested implementation.

## Scope of the lab handoff

Full CPU protocol: 30 models, 3.6m physical transitions, 300,000 training episodes, 120,000 optimizer steps, 10,500 test +600 development learned episodes, 370 rule references and 31 exact rows. Fixed budget; no checkpoint selection, early stop or automatic retry. Historical seed registry is frozen, with scope limited to declared seeds in readable local manifests and inherited registry; unseen external runs cannot be checked.

Smoke efficacy gates are null and no learning-performance claim is made. Full H1 must beat each matched control by at least 5%, win at least 8/10 paired seeds, have a Bonferroni-adjusted 97.5% CI upper below zero, and respect the nominal 10% guard. New and old FIFO cost scales are not pooled. Full execution and rsync remain human-run steps per [workflow](experiment_workflow.md).
