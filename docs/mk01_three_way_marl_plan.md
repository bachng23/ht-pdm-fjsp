# MK01 three-way MARL diagnostic plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-23
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Objective and hypotheses

This development experiment tests whether the selected
`two_specialists_x2_0` environment produces directly observable interactions
among (1) flexible-operation/routing contention, (2) technician contention,
and (3) precedence blocking during learned multi-agent control.

- H1: at least one MARL baseline encounters a three-way event in at least 1% of
  evaluation episodes and at least 20 events over its full evaluation panel.
- H2: every detected three-way event has at least one downstream operation
  unavailable solely because its predecessor is unfinished.
- H3: IQL, QMIX, and MAPPO produce measurably different objective, makespan,
  maintenance wait, and three-way-event rates under matched interaction budgets.

This is a mechanism and baseline-comparison experiment, not a final algorithm
claim.  A production conflict in the current machine-agent resolver means two
machine agents propose different routes for the same flexible operation; it is
therefore the observable routing/machine-allocation conflict.

## Fixed design

- Instance: Brandimarte MK01 HT-PdM extension.
- Technician configuration: two original specialists; all preventive and
  corrective durations multiplied by `2.0`.
- Wait semantics: `safe_noop`.
- Algorithms: cooperative IQL with independent per-machine Q networks; QMIX
  with a shared agent network and monotonic global-state mixer; independent
  actor MAPPO with centralized global critic.
- Actor information: identical local action observations for all algorithms;
  QMIX/MAPPO global state is training-only.
- Budget: 500,000 joint environment steps per model.
- Training seeds: `70000,71000,72000,73000,74000`.
- Development evaluation seeds: `61700:61900`, paired across every trained model.
- Sealed future test seeds: `62000:62100`; this experiment must not open them.

Smoke uses one training seed, 512 steps, and seeds `61990:61993`.  Smoke output
validates execution and artifact contracts only.

## Direct instrumentation

Before each joint action, count unique unfinished downstream operations with at
least one currently idle eligible machine but whose operation index exceeds the
job's next admissible operation.  After resolution, a **three-way step** is one
with both `production_conflicts > 0` and `technician_conflicts > 0`, plus a
positive precedence-blocked count.

For every evaluation step record the conflict counts, precedence-blocked count,
elapsed simulation time, maintenance-wait increment, makespan increment, and
reward.  The time/wait/makespan fields are coincident attribution, not a
counterfactual causal decomposition.  Episode rows additionally record terminal
objective and makespan and totals at three-way steps.

## Endpoints and analysis

Primary endpoint: three-way steps per 1,000 joint steps and fraction of episodes
with at least one three-way step.  Secondary endpoints: precedence-blocked
operations at three-way steps, maintenance-wait and elapsed-time increments at
those steps, terminal objective, makespan, failures, maintenance counts, and
coordination rejection rates.

Algorithm comparisons use each training seed's mean over the common 200-seed
panel, followed by paired two-sided 95% Student-t intervals.  MAPPO is the
reference.  With five training seeds these intervals are development evidence;
no multiplicity-adjusted confirmatory claim is made.

## Stopping rule and artifacts

Run every algorithm/seed cell once to its fixed budget; do not tune, early-stop,
or retry based on results.  Stop only on an exception, non-finite loss, failed
feasibility/coordination audit, or incomplete panel.  Training and evaluation
show `tqdm` progress and checkpoint after each completed cell.

Each timestamped run writes a manifest, exact base and scaled configs, episode,
decision and coordination CSVs (including partial files), summary JSON, training
curves, five recovery checkpoints and final model per cell.  Success requires
all expected cells/episodes, zero invalid or duplicate executions, at least one
baseline meeting H1, and positive precedence blocking at every three-way step.
