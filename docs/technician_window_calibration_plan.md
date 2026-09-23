# Technician-window synchronization calibration

## Material Passport

- Origin skill: ARS experiment-agent
- Artifact: preregistered executable simulation experiment plan
- Parent screen: `technician_conflict_factorial_full_cpu_20260923T195324Z`
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Execution target: CPU-only Ubuntu lab at `bachng@100.111.83.52`
- Scope: development calibration; not an RL algorithm-performance claim

## Research question and locked hypothesis

The parent factorial screen found that `Q1W1O1A1D0` creates technician conflicts
in 13% of episodes, with 78.6% of conflict units avoidable.  This experiment asks
whether synchronizing preventive-maintenance window releases can raise that
incidence into a denser learning regime without removing job precedence, changing
production routes, lengthening maintenance, or creating mainly capacity-
unavoidable congestion.

The locked directional hypothesis is that grouping machine window releases into
fewer waves increases technician-conflict incidence under independent greedy
assignment.  Moderate grouping (two or three machines per wave) is expected to
retain more avoidable coordination headroom than synchronizing all six machines.

## Fixed simulator and calibration conditions

Every condition fixes the parent factors to request persistence on, maintenance
windows on, full heterogeneous two-technician skill overlap on, technician
availability calendars on, and duration pressure off (`Q1W1O1A1D0`).  The frozen
MK01 routes, processing times, precedence, Weibull deterioration, technician
skills and durations, two maintenance rounds, eight-unit window width, and
availability intervals are unchanged.

Only the deterministic release schedule changes.  Machines are ordered
`M1,...,M6`; consecutive machines are grouped into waves of size 1, 2, 3, or 6.
Wave zero releases at times 12 and 36.  Later waves release after a fixed gap of
0.5, 1.0, or 2.0 time units.  Duplicate all-six synchronization schedules are
collapsed, producing ten candidate conditions.  The exact resolved releases are
written to the artifacts.  The original parent release schedule is retained as
an eleventh reference condition and is not eligible for selection.

Both fixed policies from the parent experiment are run on paired seeds:

- `independent_greedy`, in which each machine claims its fastest technician;
- `conflict_aware_matcher`, a maximum-cardinality minimum-duration assignment.

## Metrics

Primary calibration metric under `independent_greedy`:

- fraction of episodes with at least one technician proposal conflict.

Coordination and anti-pathology metrics:

- technician-conflict units per maintenance proposal;
- aggregate avoidable conflict fraction;
- paired conflict reduction under the matcher;
- capacity-unavoidable wait units per maintenance request onset;
- busy/off-shift claims per maintenance proposal;
- maintenance waiting time and PM-window deadline misses;
- production conflicts, makespan, objective, failures, PM and CM counts.

Audits remain hard requirements: zero invalid execution, zero duplicate operation
or technician execution, and full machine, technician and precedence feasibility.

## Seed split and stopping rule

- macOS smoke: `62300:62303` (3 seeds per condition and policy);
- full development calibration: `62400:62500` (100 seeds);
- sealed confirmation panel: `62600:62800` (200 seeds), not opened here.

Stop after exactly one full development panel.  Do not add schedules, alter gates,
or reuse the sealed seeds after inspecting the result.  No learned policy is fit,
so there are no training seeds or checkpoints.

## Locked selection rule

A non-reference condition qualifies only if all of the following hold:

1. technician conflicts occur in 20% to 40% of greedy episodes;
2. aggregate technician-conflict rate is 2% to 6% of maintenance proposals;
3. at least 70% of conflict units are avoidable at their decision epoch;
4. the matcher reduces technician-conflict units by at least 50%;
5. greedy capacity-unavoidable wait units per maintenance-request onset are no
   more than 0.35;
6. greedy busy/off-shift claims are no more than 35% of maintenance proposals;
7. both policies complete all episodes and pass every feasibility audit.

Among qualifying conditions, select the incidence closest to 30%; break ties by
higher avoidable fraction, lower capacity-unavoidable wait ratio, fewer machines
per wave, smaller inter-wave gap, then condition name.  If none qualifies, write
`selected_condition = null`.  The reference schedule is descriptive only.

Smoke success requires completion, schema checks, progress display, and audits;
it does not require a scientific candidate to qualify.

## Artifact schema

Each invocation rejects a non-empty output directory and writes:

- `benchmark_config.json`;
- `resolved_window_conditions.json`;
- `technician_window_manifest.json`;
- `technician_window_episodes.partial.csv` and final `.csv`;
- `technician_window_coordination.partial.csv` and final `.csv`;
- `technician_window_summary.json`.

The manifest records the Git commit, runtime, exact schedules, policy and
condition order, seed panels, completion status, output files, selected condition,
and global audit gate.  `tqdm` reports every condition-policy-seed episode.

## Interpretation boundary

This is a development calibration of temporal demand synchronization.  It may
nominate one environment for a separately preregistered confirmation and later
MARL training.  It cannot establish that QMIX, QPLEX, or another learned policy
will exploit the measured coordination headroom, and it does not estimate a
general causal effect outside the enumerated release schedules.
