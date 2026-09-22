# MK01 joint technician-conflict pilot

## Material Passport

- Artifact: executable code-experiment plan
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Architecture under test: six machine agents with deterministic joint-action resolution
- Status: implementation and local smoke validation
- Scope: development diagnostic; not an algorithm-performance or final-test result

## Locked question and hypothesis

This pilot asks whether the frozen six-machine, two-technician MK01 extension
creates observable technician contention when independent machine agents act at
the same decision epoch.

The locked hypothesis is that the configured threshold policy (`0.18`) produces
a non-degenerate but manageable technician-conflict signal, and that lowering
the policy threshold to `0.05` increases maintenance pressure and technician
conflicts. The production routes, processing times, degradation parameters,
technician skills, and service times remain unchanged.

## Conditions

All conditions use deterministic per-machine shortest-processing-time routing,
corrective maintenance priority, and the same stochastic seed panel.

1. `reactive_joint_spt`: preventive maintenance is never selected voluntarily.
2. `threshold_018_joint_spt`: select the fastest feasible preventive action
   when the selected production action has failure probability at least `0.18`.
3. `threshold_005_joint_spt`: identical rule with threshold `0.05`, used only
   as a maintenance-pressure stress condition.

Agents choose independently. The existing CTDE resolver rotates agent priority,
accepts a conflict-free subset, and records rejected production and technician
proposals. If the environment mask makes `wait` invalid while preventive
maintenance is the only feasible local action, that selection is retained but
recorded separately as `forced_preventive`; it is not counted as voluntary
threshold behavior. No learned policy or centralized pre-coordination is used.

## Metrics

Primary mechanism metrics:

- total technician conflicts;
- fraction of episodes with at least one technician conflict;
- technician-conflict rate per maintenance proposal;
- total maintenance proposals.
- threshold-triggered versus action-mask-forced preventive proposals.

Secondary metrics:

- total and per-episode rejection rate;
- production conflicts;
- accepted proposals and waits;
- makespan, objective, failures, preventive and corrective maintenance;
- invalid executions and duplicate operation/technician executions.

The independent unit is one stochastic environment seed. Conditions are paired
by seed. This pilot reports descriptive paired effects and confidence intervals
only after the locked gate is evaluated.

## Seed split

- Local macOS smoke: `61400:61405` (five seeds per condition).
- Full Ubuntu/server diagnostic: `61500:61700` (200 seeds per condition).
- Seeds `62000:62100` remain unopened for a future confirmatory configuration
  check if the maintenance overlay is changed.

## Locked gate and stopping rule

The current maintenance overlay supports MARL coordination experiments only if
all of the following pass on the full diagnostic panel:

1. all 600 episodes terminate and all feasibility audits pass;
2. invalid and duplicate executions remain zero in every condition;
3. `threshold_018_joint_spt` has technician conflicts in at least 10% of seeds;
4. its aggregate technician-conflict rate is at least 1% of maintenance proposals;
5. its aggregate rejected-proposal rate is at most 20% of all proposals;
6. the `0.05` stress condition has at least as many maintenance proposals and
   technician conflicts as the `0.18` condition.

Stop after one full panel. A failed gate triggers maintenance-overlay redesign
on development seeds; it does not authorize opening the reserved panel or
starting expensive MARL training.

## Artifact schema

Each invocation requires a new UTC timestamped output directory and writes:

- `benchmark_config.json`;
- `technician_conflict_manifest.json`;
- `technician_conflict_episodes.csv` and `.partial.csv`;
- `technician_conflict_decisions.csv` and `.partial.csv`;
- `technician_conflict_coordination.csv` and `.partial.csv`;
- `technician_conflict_summary.json`.

The CLI displays a `tqdm` progress bar over every condition-seed episode.

## Rented Ubuntu server command

Run from the local Mac. This diagnostic is simulator-bound and does not require
CUDA even when a GPU is available.

```bash
ssh -p 20002 root@194.93.49.15 'export PATH="/root/.local/bin:$PATH" && cd /root/ht-pdm-fjsp && git fetch origin && (git switch codex/mk01-technician-conflict-pilot || git switch --track origin/codex/mk01-technician-conflict-pilot) && git pull --ff-only && uv sync --frozen && set -o pipefail && RUN_ID="mk01_technician_conflicts_$(date -u +%Y%m%dT%H%M%SZ)" && mkdir -p "artifacts/$RUN_ID" && uv run ht-pdm-fjsp-technician-conflicts --profile full --output-dir "artifacts/$RUN_ID" 2>&1 | tee "artifacts/$RUN_ID.log"'
```

Pull every artifact back to the local Mac with partial-transfer preservation and
progress display:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && rsync -avhP --partial -e "ssh -p 20002" root@194.93.49.15:/root/ht-pdm-fjsp/artifacts/ "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
